local _ = require 'moses'
local paths = require 'paths'
local classic = require 'classic'
local nn = require 'nn'
local nninit = require 'nninit'
local image = require 'image'
local DuelAggregator = require 'modules/DuelAggregator'
require 'classic.torch' -- Enables serialisation
require 'rnn'
require 'dpnn' -- Adds gradParamClip method
require 'modules/GuidedReLU'
require 'modules/DeconvnetReLU'
require 'modules/GradientRescale'
--nn.FastLSTM.usenngraph = true -- Use faster FastLSTM TODO: Re-enable once nngraph #109 is resolved

local Model = classic.class('Model')

-- Creates a Model (a helper for the network it creates)
function Model:_init(opt)
  -- Extract relevant options
  self.tensorType = opt.tensorType
  self.gpu = opt.gpu
  self.colorSpace = opt.colorSpace
  self.width = opt.width
  self.height = opt.height
  self.modelBody = opt.modelBody
  self.hiddenSize = opt.hiddenSize
  self.histLen = opt.histLen
  self.duel = opt.duel
  self.bootstraps = opt.bootstraps
  self.recurrent = opt.recurrent
  self.env = opt.env
  self.async = opt.async
  self.a3c = opt.async == 'A3C'
  self.stateSpec = opt.stateSpec
  
  self.m = opt.actionSpec[3][2] - opt.actionSpec[3][1] + 1 -- Number of discrete actions
  -- Set up resizing
  if opt.width ~= 0 or opt.height ~= 0 then
    self.resize = true
    self.width = opt.width ~= 0 and opt.width or opt.stateSpec[2][3]
    self.height = opt.height ~= 0 and opt.height or opt.stateSpec[2][2]
  end
end

-- Processes a single frame for DQN input; must not return same memory to prevent side-effects
function Model:preprocess(observation)
  local frame = observation:type(self.tensorType) -- Convert from CudaTensor if necessary
  
  -- Perform colour conversion if needed
  if self.colorSpace then
    frame = image['rgb2' .. self.colorSpace](frame)
  end
  
  -- Resize screen if needed
  if self.resize then
    frame = image.scale(frame, self.width, self.height)
  end

  -- Clone if needed
  if frame == observation then
    frame = frame:clone()
  end

  return frame
end

-- Creates a DQN/AC model body
function Model:createBody()
  -- Number of input frames for recurrent networks is always 1
  local histLen = self.recurrent and 1 or self.histLen
  local net
  
  if paths.filep(self.modelBody .. '.lua') then
    net = require(self.modelBody)
    net:type(self.tensorType)
  elseif self.env == 'rlenvs.Atari' then
    net = nn.Sequential()
    net:add(nn.View(histLen*self.stateSpec[2][1], self.stateSpec[2][2], self.stateSpec[2][3])) -- Concatenate history in channel dimension
    net:add(nn.SpatialConvolution(histLen*self.stateSpec[2][1], 32, 8, 8, 4, 4, 1, 1))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialConvolution(32, 64, 4, 4, 2, 2))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1))
    net:add(nn.ReLU(true))
  else -- Default network/Catch network
    net = nn.Sequential()
    net:add(nn.View(histLen*self.stateSpec[2][1], self.stateSpec[2][2], self.stateSpec[2][3]))
    net:add(nn.SpatialConvolution(histLen*self.stateSpec[2][1], 32, 5, 5, 2, 2, 1, 1))
    net:add(nn.ReLU(true))
    net:add(nn.SpatialConvolution(32, 32, 5, 5, 2, 2))
    net:add(nn.ReLU(true))
  end

  return net
end

-- Calculates network output size
local function getOutputSize(net, inputDims)
  return net:forward(torch.Tensor(torch.LongStorage(inputDims))):size():totable()
end

-- Creates a DQN/AC model based on a number of discrete actions
function Model:create()
  -- Number of input frames for recurrent networks is always 1
  local histLen = self.recurrent and 1 or self.histLen

  -- Network starting with convolutional layers/model body
  local net = nn.Sequential()
  if self.recurrent then
    net:add(nn.Copy(nil, nil, true)) -- Needed when splitting batch x seq x input over seq for DRQN; better than nn.Contiguous
  end
  
  -- Add network body
  net:add(self:createBody())
  -- Calculate body output size
  local bodyOutputSize = torch.prod(torch.Tensor(getOutputSize(net, _.append({histLen}, self.stateSpec[2]))))
  net:add(nn.View(bodyOutputSize))

  -- Network head
  local head = nn.Sequential()
  local heads = math.max(self.bootstraps, 1)
  if self.duel then
    -- Value approximator V^(s)
    local valStream = nn.Sequential()
    if self.recurrent then
      local lstm = nn.FastLSTM(bodyOutputSize, self.hiddenSize, self.histLen)
      lstm.i2g:init({'bias', {{3*self.hiddenSize+1, 4*self.hiddenSize}}}, nninit.constant, 1)
      valStream:add(lstm)
    else
      valStream:add(nn.Linear(bodyOutputSize, self.hiddenSize))
      valStream:add(nn.ReLU(true))
    end
    valStream:add(nn.Linear(self.hiddenSize, 1)) -- Predicts value for state

    -- Advantage approximator A^(s, a)
    local advStream = nn.Sequential()
    if self.recurrent then
      local lstm = nn.FastLSTM(bodyOutputSize, self.hiddenSize, self.histLen)
      lstm.i2g:init({'bias', {{3*self.hiddenSize+1, 4*self.hiddenSize}}}, nninit.constant, 1)
      advStream:add(lstm)
    else
      advStream:add(nn.Linear(bodyOutputSize, self.hiddenSize))
      advStream:add(nn.ReLU(true))
    end
    advStream:add(nn.Linear(self.hiddenSize, self.m)) -- Predicts action-conditional advantage

    -- Streams container
    local streams = nn.ConcatTable()
    streams:add(valStream)
    streams:add(advStream)

    -- Network finishing with fully connected layers
    head:add(nn.GradientRescale(1/math.sqrt(2), true)) -- Heuristic that mildly increases stability for duel
    -- Create dueling streams
    head:add(streams)
    -- Add dueling streams aggregator module
    head:add(DuelAggregator(self.m))
  else
    if self.recurrent then
      local lstm = nn.FastLSTM(bodyOutputSize, self.hiddenSize, self.histLen)
      lstm.i2g:init({'bias', {{3*self.hiddenSize+1, 4*self.hiddenSize}}}, nninit.constant, 1) -- Extra: high forget gate bias (Gers et al., 2000)
      head:add(lstm)
      if self.async then
        lstm:remember('both')
      end
    else
      head:add(nn.Linear(bodyOutputSize, self.hiddenSize))
      head:add(nn.ReLU(true)) -- DRQN paper reports worse performance with ReLU after LSTM
    end
    head:add(nn.Linear(self.hiddenSize, self.m)) -- Note: Tuned DDQN uses shared bias at last layer
  end

  if self.bootstraps > 0 then
    -- Add bootstrap heads
    local headConcat = nn.ConcatTable()
    for h = 1, heads do
      -- Clone head structure
      local bootHead = head:clone()
      -- Each head should use a different random initialisation to construct bootstrap (currently Torch default)
      local linearLayers = bootHead:findModules('nn.Linear')
      for l = 1, #linearLayers do
        linearLayers[l]:init('weight', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)}):init('bias', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)})
      end
      headConcat:add(bootHead)
    end
    net:add(nn.GradientRescale(1/self.bootstraps)) -- Normalise gradients by number of heads
    net:add(headConcat)
  elseif self.a3c then
    -- Actor-critic does not use the normal head but instead a concatenated value function V and policy π
    net:add(nn.Linear(bodyOutputSize, self.hiddenSize))
    net:add(nn.ReLU(true))

    local valueAndPolicy = nn.ConcatTable() -- π and V share all layers except the last

    -- Value function V(s; θv)
    local valueFunction = nn.Linear(self.hiddenSize, 1)

    -- Policy π(a | s; θπ)
    local policy = nn.Sequential()
    policy:add(nn.Linear(self.hiddenSize, self.m))
    policy:add(nn.SoftMax())

    valueAndPolicy:add(valueFunction)
    valueAndPolicy:add(policy)

    net:add(valueAndPolicy)
  else
    -- Add head via ConcatTable (simplifies bootstrap code in agent)
    local headConcat = nn.ConcatTable()
    headConcat:add(head)
    net:add(headConcat)
  end

  if not self.a3c then
    net:add(nn.JoinTable(1, 1))
    net:add(nn.View(heads, self.m))

    if not self.async and self.recurrent then
      local sequencer = nn.Sequencer(net)
      sequencer:remember('both') -- Keep hidden state between forward calls; requires manual calls to forget
      net = nn.Sequential():add(nn.SplitTable(1, #self.stateSpec[2] + 1)):add(sequencer):add(nn.SelectTable(-1))
    end
  end

  -- GPU conversion
  if self.gpu > 0 then
    require 'cunn'
    net:cuda()
  end

  -- Save reference to network
  self.net = net

  return net
end

function Model:setNetwork(net)
  self.net = net
end

-- Return list of convolutional filters as list of images
function Model:getFilters()
  local filters = {}

  -- Find convolutional layers
  local convs = self.net:findModules('nn.SpatialConvolution')
  for i, v in ipairs(convs) do
    -- Add filter to list (with each layer on a separate row)
    filters[#filters + 1] = image.toDisplayTensor(v.weight:view(v.nOutputPlane*v.nInputPlane, v.kH, v.kW), 1, v.nInputPlane, true)
  end

  return filters
end

-- Set ReLUs up for specified saliency visualisation type
function Model:setSaliency(saliency)
  -- Set saliency
  self.saliency = saliency

  -- Find ReLUs on existing model
  local relus, relucontainers = self.net:findModules('nn.ReLU')
  if #relus == 0 then
    relus, relucontainers = self.net:findModules('nn.GuidedReLU')
  end
  if #relus == 0 then
    relus, relucontainers = self.net:findModules('nn.DeconvnetReLU')
  end

  -- Work out which ReLU to use now
  local layerConstructor = nn.ReLU
  self.relus = {} --- Clear special ReLU list to iterate over for salient backpropagation
  if saliency == 'guided' then
    layerConstructor = nn.GuidedReLU
  elseif saliency == 'deconvnet' then
    layerConstructor = nn.DeconvnetReLU
  end

  -- Replace ReLUs
  for i = 1, #relus do
    -- Create new special ReLU
    local layer = layerConstructor()

    -- Copy everything over
    for key, val in pairs(relus[i]) do
      layer[key] = val
    end

    -- Find ReLU in containing module and replace
    for j = 1, #(relucontainers[i].modules) do
      if relucontainers[i].modules[j] == relus[i] then
        relucontainers[i].modules[j] = layer
      end
    end
  end

  -- Create special ReLU list to iterate over for salient backpropagation
  self.relus = self.net:findModules(saliency == 'guided' and 'nn.GuidedReLU' or 'nn.DeconvnetReLU')
end

-- Switches the backward computation of special ReLUs for salient backpropagation
function Model:salientBackprop()
  for i, v in ipairs(self.relus) do
    v:salientBackprop()
  end
end

-- Switches the backward computation of special ReLUs for normal backpropagation
function Model:normalBackprop()
  for i, v in ipairs(self.relus) do
    v:normalBackprop()
  end
end

return Model
