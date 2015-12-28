local _ = require 'moses'
local nn = require 'nn'
require 'modules/GradientRescale'
local image = require 'image'

local model = {}

-- Adds a cuDNN module if available (waiting on https://github.com/soumith/cudnn.torch/pull/76)
-- TODO: Use fbcunn if available since FFT is faster for large convs
local toCuDNN = function(mod, ...)
  if cudnn then
    if mod == 'relu' then
      return cudnn.ReLU(true)
    elseif mod == 'conv' then
      return cudnn.SpatialConvolution(...)
    end
  else
    if mod == 'relu' then
      return nn.ReLU(true)
    elseif mod == 'conv' then
      return nn.SpatialConvolution(...)
    end
  end
end

-- Calculates the output size of a network (returns LongStorage)
local calcOutputSize = function(network, inputSize)
  if cudnn then
    return network:cuda():forward(torch.CudaTensor(inputSize)):size()
  else
    return network:forward(torch.Tensor(inputSize)):size()
  end
end

-- Processes the full screen for DQN input
model.preprocess = function(observation, opt)
  local input = torch.Tensor(observation:size(1), opt.nChannels, opt.height, opt.width)
  if opt.gpu > 0 then
    input = input:cuda()
  end

  -- Loop over received frames
  for f = 1, observation:size(1) do
    -- Load frame
    local frame = observation:select(1, f):float() -- image does not work with CudaTensor
    -- Perform colour conversion
    if opt.colorSpace ~= 'rgb' then
      frame = image['rgb2' .. opt.colorSpace](frame)
    end
    -- Resize 210x160 screen
    input[{{f}, {}, {}, {}}] = image.scale(frame, opt.width, opt.height)
  end

  return input
end

-- TODO: Work out how to get 4 observations
-- Creates a dueling DQN
model.create = function(A, opt)
  -- Use cuDNN if available
  pcall(require, 'cudnn')
  if opt.gpu == 0 then
    cudnn = nil
  end

  -- Number of discrete actions
  local m = _.size(A) 

  -- Network starting with convolutional layers
  local net = nn.Sequential()
  net:add(toCuDNN('conv', opt.nChannels, 32, 8, 8, 4, 4))
  net:add(toCuDNN('relu'))
  net:add(toCuDNN('conv', 32, 64, 4, 4, 2, 2))
  net:add(toCuDNN('relu'))
  net:add(toCuDNN('conv', 64, 64, 3, 3, 1, 1))
  net:add(toCuDNN('relu'))
  -- Calculate convolutional network output size
  local convOutputSize = torch.prod(torch.Tensor(calcOutputSize(net, torch.LongStorage({opt.nChannels, opt.height, opt.width})):totable()))

  -- Value approximator V^(s)
  local valStream = nn.Sequential()
  valStream:add(nn.Linear(convOutputSize, 512))
  valStream:add(toCuDNN('relu'))
  valStream:add(nn.Linear(512, 1)) -- Predicts value for state

  -- Advantage approximator A^(s, a)
  local advStream = nn.Sequential()
  advStream:add(nn.Linear(convOutputSize, 512))
  advStream:add(toCuDNN('relu'))
  advStream:add(nn.Linear(512, m)) -- Predicts action-conditional advantage

  -- Streams container
  local streams = nn.ConcatTable()
  streams:add(valStream)
  streams:add(advStream)
  
  -- Aggregator module
  local aggregator = nn.Sequential()
  local aggParallel = nn.ParallelTable()
  -- Value duplicator (for each action)
  local valDuplicator = nn.Sequential()
  local valConcat = nn.ConcatTable()
  for a = 1, m do
    valConcat:add(nn.Identity())
  end
  valDuplicator:add(valConcat)
  valDuplicator:add(nn.JoinTable(1, 1))
  -- Add value duplicator
  aggParallel:add(valDuplicator)
  -- Advantage duplicator (for calculating and subtracting mean)
  local advDuplicator = nn.Sequential()
  local advConcat = nn.ConcatTable()
  advConcat:add(nn.Identity())
  -- Advantage mean duplicator
  local advMeanDuplicator = nn.Sequential()
  advMeanDuplicator:add(nn.Mean(1, 1))
  local advMeanConcat = nn.ConcatTable()
  for a = 1, m do
    advMeanConcat:add(nn.Identity())
  end
  advMeanDuplicator:add(advMeanConcat)
  advMeanDuplicator:add(nn.JoinTable(1, 1))
  advConcat:add(advMeanDuplicator)
  advDuplicator:add(advConcat)
  -- Subtract mean from advantage values
  advDuplicator:add(nn.CSubTable())
  aggParallel:add(advDuplicator)
  -- Calculate Q^ from V^ and A^
  aggregator:add(aggParallel)
  aggregator:add(nn.CAddTable())

  -- Network finishing with fully connected layers
  net:add(nn.View(convOutputSize))
  net:add(nn.GradientRescale(1 / math.sqrt(2))) -- Heuristic that mildly increases stability for duel
  -- Create dueling streams
  net:add(streams)
  -- Join dueling streams
  net:add(aggregator)

  if opt.gpu > 0 then
    require 'cunn'
    net:cuda()
  end

  return net
end

return model
