local _ = require 'moses'
local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
local image = require 'image'

local model = {}

-- Processes the full screen for DQN input
local preprocess = function(observation)
  local input = torch.CudaTensor(observation:size(1), 1, 84, 84)

  -- Loop over received frames
  for f = 1, observation:size(1) do
    -- Convert to grayscale
    local frame = image.rgb2y(observation:select(1, f):float()) -- image does not work with CudaTensor
    -- Reduce 210x160 screen to 84x84
    input[{{f}, {}, {}, {}}] = image.scale(frame, 84, 84)
  end

  return input
end

-- Creates a DQN model
local createModel = function(A)
  local net = nn.Sequential()
  -- TODO: Work out how to get 4 observations
  net:add(cudnn.SpatialConvolution(1, 32, 8, 8, 4, 4))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(32, 64, 4, 4, 2, 2))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1))
  net:add(cudnn.ReLU(true))
  net:add(nn.View(64*7*7))
  net:add(nn.Linear(64*7*7, 512))
  net:add(cudnn.ReLU(true))
  net:add(nn.Linear(512, _.size(A)))
  net:cuda()

  return net
end

-- An agent must observe, act, and learn

-- Creates an agent from a DQN
model.createAgent = function(gameEnv, opt)
  local agent = {}
  local gameActions = gameEnv:getActions()
  local net = createModel(gameActions)
  
  --[[
  -- Create transition table.
  ---- assuming the transition table always gets floating point input
  ---- (Foat or Cuda tensors) and always returns one of the two, as required
  ---- internally it always uses ByteTensors for states, scaling and
  ---- converting accordingly
  local transition_args = {
    stateDim = self.state_dim, numActions = self.n_actions,
    histLen = self.hist_len, gpu = self.gpu,
    maxSize = self.replay_memory, histType = self.histType,
    histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
    bufferSize = self.bufferSize
  }

  self.transitions = dqn.TransitionTable(transition_args)
  --]]

  -- Returns an action index given an observation
  agent.observe = function(self, observation)
    local __, ind = torch.max(net:forward(preprocess(observation)), 1)
    return ind[1]
  end

  -- Performs an action on the environment
  agent.act = function(self, actionIndex, trainMode)
    -- Returns screen, reward, terminal
    return gameEnv:step(gameActions[actionIndex], trainMode)
  end

  -- TODO: Perform a learning step
  agent.learn = function(self, reward)
  end

  return agent
end

return model
