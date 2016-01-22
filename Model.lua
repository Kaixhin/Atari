local _ = require 'moses'
local classic = require 'classic'
local nn = require 'nn'
local image = require 'image'
local DuelAggregator = require 'modules/DuelAggregator'
require 'classic.torch' -- Enables serialisation
require 'dpnn' -- Adds gradParamClip method
require 'modules/GradientRescale'

local Model = classic.class('Model')

-- Creates a Model (a helper for the network it creates)
function Model:_init(opt)
  -- Extract relevant options
  self.gpu = opt.gpu
  self.colorSpace = opt.colorSpace
  self.width = opt.width
  self.height = opt.height
  self.nChannels = opt.nChannels
  self.histLen = opt.histLen
  self.duel = opt.duel
  self.ale = opt.ale

  -- Get cuDNN if available
  self.hasCudnn = pcall(require, 'cudnn')
end

-- Returns optimal module based on type
function Model:bestModule(mod, ...)
  if mod == 'relu' then
    if self.gpu > 0 and self.hasCudnn then
      return cudnn.ReLU(...)
    else
      return nn.ReLU(...)
    end
  elseif mod == 'conv' then
    if self.gpu > 0 and self.hasCudnn then
      return cudnn.SpatialConvolution(...)
    else
      return nn.SpatialConvolution(...)
    end
  end
end

-- Calculates the output size of a network (returns LongStorage)
function Model:calcOutputSize(network, inputSizes)
  if self.gpu > 0 and cudnn then
    return network:cuda():forward(torch.CudaTensor(torch.LongStorage(inputSizes))):size()
  else
    return network:forward(torch.Tensor(torch.LongStorage(inputSizes))):size()
  end
end

-- Processes a single frame for DQN input
function Model:preprocess(observation)
  if self.ale then
    -- Load frame
    local frame = observation:select(1, 1):float() -- Convert from CudaTensor if necessary
    -- Perform colour conversion
    if self.colorSpace ~= 'rgb' then
      frame = image['rgb2' .. self.colorSpace](frame)
    end

    -- Resize 210x160 screen
    return image.scale(frame, self.width, self.height)
  else
    -- Return normal Catch screen
    return observation
  end
end

-- Creates a dueling DQN based on a number of discrete actions
function Model:create(m)
  -- Size of fully connected layers
  local hiddenSize = self.ale and 512 or 32

  -- Network starting with convolutional layers
  local net = nn.Sequential()
  net:add(nn.View(self.histLen*self.nChannels, self.height, self.width)) -- Concatenate history in channel dimension
  if self.ale then
    net:add(self:bestModule('conv', self.histLen*self.nChannels, 32, 8, 8, 4, 4))
    net:add(self:bestModule('relu', true))
    net:add(self:bestModule('conv', 32, 64, 4, 4, 2, 2))
    net:add(self:bestModule('relu', true))
    net:add(self:bestModule('conv', 64, 64, 3, 3, 1, 1))
    net:add(self:bestModule('relu', true))
  else
    net:add(self:bestModule('conv', self.histLen*self.nChannels, 8, 5, 5, 2, 2, 2, 2))
    net:add(self:bestModule('relu', true))
    net:add(self:bestModule('conv', 8, 16, 3, 3, 1, 1, 1, 1))
    net:add(self:bestModule('relu', true))
  end
  -- Calculate convolutional network output size
  local convOutputSize = torch.prod(torch.Tensor(self:calcOutputSize(net, {self.histLen*self.nChannels, self.height, self.width}):totable()))
  net:add(nn.View(convOutputSize))

  if self.duel then
    -- Value approximator V^(s)
    local valStream = nn.Sequential()
    valStream:add(nn.Linear(convOutputSize, hiddenSize))
    valStream:add(self:bestModule('relu', true))
    valStream:add(nn.Linear(hiddenSize, 1)) -- Predicts value for state

    -- Advantage approximator A^(s, a)
    local advStream = nn.Sequential()
    advStream:add(nn.Linear(convOutputSize, hiddenSize))
    advStream:add(self:bestModule('relu', true))
    advStream:add(nn.Linear(hiddenSize, m)) -- Predicts action-conditional advantage

    -- Streams container
    local streams = nn.ConcatTable()
    streams:add(valStream)
    streams:add(advStream)
    
    -- Network finishing with fully connected layers
    net:add(nn.GradientRescale(1 / math.sqrt(2), true)) -- Heuristic that mildly increases stability for duel
    -- Create dueling streams
    net:add(streams)
    -- Add dueling streams aggregator module
    net:add(DuelAggregator(m))
  else
    net:add(nn.Linear(convOutputSize, hiddenSize))
    net:add(self:bestModule('relu', true))
    net:add(nn.Linear(hiddenSize, m))
  end

  if self.gpu > 0 then
    require 'cunn'
    net:cuda()
  end

  return net
end

return Model
