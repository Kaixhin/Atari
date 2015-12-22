local _ = require 'moses'
local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
local image = require 'image'

local model = {}

-- Creates a DQN model
local createModel = function(A)
  local net = nn.Sequential()
  -- TODO: Work out how to get 4 observations
  net:add(cudnn.SpatialConvolution(1, 32, 8, 8, 4, 4))
  net:add(cudnn.ReLU(true))
  -- TODO: Check if 2x2 MaxPooling needed
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
model.createAgent = function(A)
  local agent = {}
  local net = createModel(A)

  -- Returns an action given an observation
  agent.observe = function(self, observation)
    local input = torch.CudaTensor(observation:size(1), 1, 84, 84)

    -- Loop over received frames
    for f = 1, observation:size(1) do
      -- Convert to grayscale
      local frame = image.rgb2y(observation:select(1, f))
      -- Reduce 210x160 screen to 84x84
      input[{{f}, {}, {}, {}}] = image.scale(frame, 84, 84)
    end

    local __, ind = torch.max(net:forward(input), 1)
    return ind[1]
  end

  -- Performs an action on the environment???
  agent.act = function(self)
  end

  -- TODO: Perform a learning step
  agent.learn = function(self)
  end

  return agent
end

return model
