local _ = require 'moses'
local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
local image = require 'image'
local experience = require 'experience'

local model = {}

-- Processes the full screen for DQN input
local preprocess = function(observation)
  local input = torch.CudaTensor(observation:size(1), 1, 84, 84)

  -- Loop over received frames
  for f = 1, observation:size(1) do
    -- Convert to grayscale
    local frame = image.rgb2y(observation:select(1, f):float()) -- image does not work with CudaTensor
    -- Resize 210x160 screen to 84x84
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

-- Creates an agent from a DQN
model.createAgent = function(gameEnv, opt)
  local agent = {}
  local A = gameEnv:getActions()
  -- DQN
  local net = createModel(A)
  -- Model parameters θ
  local theta, dTheta = net:getParameters()
  -- Experience replay memory
  local memory = experience.createMemory(opt.expReplMem, {1, 84, 84})
  -- SARSA
  local s0 = nil
  local a0 = nil
  local r0 = nil
  local s1 = nil
  local a1 = nil
  
  -- TODO: If transition is terminal then TD error = reward
  local learnFromTuples = function(states, actions, rewards, transitions)
    -- Calculate max Q-value from next state
    local QMax = torch.max(net:forward(transitions), 2)
    -- Calculate target Y
    local Y = torch.add(rewards, torch.mul(QMax, opt.gamma)) -- TODO: Add target network and Double Q calculation

    -- Get all predicted Q-values from the current state
    local QCurr = net:forward(states)
    local QTaken = torch.CudaTensor(opt.batchSize)
    -- Get prediction of current Q-value with given actions
    for q = 1, opt.batchSize do
      QTaken[q] = QCurr[q][actions[q]]
    end

    -- Calculate TD error
    local tdErr = QTaken - Y
    -- Clamp TD error (approximates Huber loss)
    tdErr:clamp(-opt.tdClamp, opt.tdClamp)
    
    -- Zero QCurr outputs (no error)
    QCurr:zero()
    -- Set TD errors with given actions
    for q = 1, opt.batchSize do
      QCurr[q][actions[q]] = tdErr[q]
    end

    -- Reset gradients
    dTheta:zero()
    -- Backpropagate loss
    net:backward(states, QCurr)

    -- Update parameters
    theta:add(torch.mul(dTheta, opt.alpha))
    -- TODO: Too much instability -> NaNs
  end

  -- Outputs an action (index) to perform on the environment
  agent.observe = function(self, observation, mode)
    local state = preprocess(observation)
    local aIndex
    
    -- Choose action by ε-greedy exploration
    if mode == 'test' or math.random() < opt.epsilon[opt.step] then 
      aIndex = torch.random(1, _.size(A))
    else
      local __, ind = torch.max(net:forward(state), 1)
      aIndex = ind[1]
    end

    -- SARSA update
    s0 = s1
    a0 = a1
    s1 = state
    a1 = aIndex

    return aIndex
  end

  -- Perform a learning step from the latest reward
  agent.learn = function(self, reward)
    if a0 ~= nil and opt.alpha > 0 then
      -- Store experience
      memory.store(s0, a0, r0, s1)

      -- Occasionally sample from replay memory
      if opt.step % opt.memSampleFreq == 0 and memory.size() >= opt.batchSize then
        -- Sample uniformly
        local indices = torch.randperm(memory.size()):long()
        indices = indices[{{1, opt.batchSize}}]
        -- Learn
        learnFromTuples(memory.retrieve(indices))
      end
    end

    r0 = reward
  end

  return agent
end

return model
