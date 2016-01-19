local _ = require 'moses'
local nn = require 'nn'
local image = require 'image'
require 'modules/GradientRescale'
local DuelAggregator = require 'modules/DuelAggregator'

local model = {}

-- Returns optimal module based on type
local bestModule = function(mod, ...)
  if mod == 'relu' then
    if model.gpu > 0 and model.hasCudnn then
      return cudnn.ReLU(...)
    else
      return nn.ReLU(...)
    end
  elseif mod == 'conv' then
    if model.gpu > 0 and model.hasCudnn then
      return cudnn.SpatialConvolution(...)
    else
      return nn.SpatialConvolution(...)
    end
  end
end

-- Calculates the output size of a network (returns LongStorage)
local calcOutputSize = function(network, inputSizes)
  if model.gpu > 0 and cudnn then
    return network:cuda():forward(torch.CudaTensor(torch.LongStorage(inputSizes))):size()
  else
    return network:forward(torch.Tensor(torch.LongStorage(inputSizes))):size()
  end
end

-- Processes a single frame for DQN input
model.preprocess = function(observation)
  if model.ale then
    -- Load frame
    model.buffers.frame = observation:select(1, 1):float() -- Convert from CudaTensor if necessary
    -- Perform colour conversion
    if model.colorSpace ~= 'rgb' then
      model.buffers.convertedFrame = image['rgb2' .. model.colorSpace](model.buffers.frame)
    end

    -- Resize 210x160 screen
    return image.scale(model.buffers.convertedFrame, model.width, model.height)
  else
    -- Return normal Catch screen
    return observation
  end
end

-- Creates a dueling DQN based on a number of discrete actions
model.create = function(m)
  -- Size of fully connected layers
  local hiddenSize = model.ale and 512 or 32

  -- Network starting with convolutional layers
  local net = nn.Sequential()
  net:add(nn.View(model.histLen*model.nChannels, model.height, model.width)) -- Concatenate history in channel dimension
  if model.ale then
    net:add(bestModule('conv', model.histLen*model.nChannels, 32, 8, 8, 4, 4))
    net:add(bestModule('relu', true))
    net:add(bestModule('conv', 32, 64, 4, 4, 2, 2))
    net:add(bestModule('relu', true))
    net:add(bestModule('conv', 64, 64, 3, 3, 1, 1))
    net:add(bestModule('relu', true))
  else
    net:add(bestModule('conv', model.histLen*model.nChannels, 16, 3, 3, 2, 2, 1, 1))
    net:add(bestModule('relu', true))
    net:add(bestModule('conv', 16, 32, 3, 3, 1, 1, 1, 1))
    net:add(bestModule('relu', true))
  end
  -- Calculate convolutional network output size
  local convOutputSize = torch.prod(torch.Tensor(calcOutputSize(net, {model.histLen*model.nChannels, model.height, model.width}):totable()))
  net:add(nn.View(convOutputSize))

  if model.duel then
    -- Value approximator V^(s)
    local valStream = nn.Sequential()
    valStream:add(nn.Linear(convOutputSize, hiddenSize))
    valStream:add(bestModule('relu', true))
    valStream:add(nn.Linear(hiddenSize, 1)) -- Predicts value for state

    -- Advantage approximator A^(s, a)
    local advStream = nn.Sequential()
    advStream:add(nn.Linear(convOutputSize, hiddenSize))
    advStream:add(bestModule('relu', true))
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
    net:add(bestModule('relu', true))
    net:add(nn.Linear(hiddenSize, m))
  end

  if model.gpu > 0 then
    require 'cunn'
    net:cuda()
  end

  return net
end

-- Initialises model
model.init = function(opt)
  -- Extract relevant options
  model.gpu = opt.gpu
  model.colorSpace = opt.colorSpace
  model.width = opt.width
  model.height = opt.height
  model.nChannels = opt.nChannels
  model.histLen = opt.histLen
  model.duel = opt.duel
  model.ale = opt.ale

  -- Create "buffers"
  if opt.ale then
    model.buffers = {
      frame = torch.FloatTensor(opt.origChannels, opt.origHeight, opt.origWidth),
      convertedFrame = torch.FloatTensor(model.nChannels, opt.origHeight, opt.origWidth)
    }
  end

  -- Get cuDNN if available
  model.hasCudnn = pcall(require, 'cudnn')
end

return model
