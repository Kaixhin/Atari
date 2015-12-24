local _ = require 'moses'
local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
local image = require 'image'
local optim = require 'optim'
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
  -- Model parameters θ and gradients dθ
  local theta, dTheta = net:getParameters()
  -- Experience replay memory
  local memory = experience.createMemory(opt.memSize, {1, 84, 84})
  -- Training mode
  agent.isTraining = false
  -- Experience variables for learning
  local state -- Updated on observe
  local action, reward, transition, terminal -- Updated on act
  -- Optimiser parameters
  local optimParams = {
    learningRate = opt.alpha,
    alpha = opt.momentum -- TODO: Select proper momentum for optimisers other than RMSprop
  }

  -- Sets training mode
  agent.training = function(self)
    self.isTraining = true
  end

  -- Sets evaluation mode
  agent.evaluate = function(self)
    self.isTraining = false
  end
  
  -- Outputs an action (index) to perform on the environment
  agent.observe = function(self, observation)
    local s = preprocess(observation)
    local aIndex

    if self.isTraining then
      -- If training, choose action by ε-greedy exploration
      if math.random() < opt.epsilon[opt.step] then 
        aIndex = torch.random(1, _.size(A))
      else
        local __, ind = torch.max(net:forward(s), 1)
        aIndex = ind[1]
      end

      -- Store state
      state = s
    else
      -- If not training, choose action greedily
      local __, ind = torch.max(net:forward(s), 1)
      aIndex = ind[1]
    end

    return aIndex
  end

  -- Acts on the environment (and can learn)
  agent.act = function(self, aIndex)
    local scr, rew, term = gameEnv:step(A[aIndex], self.isTraining)

    -- If training
    if self.isTraining then
      -- Store action (index), reward, transition and terminal
      action, reward, transition, terminal = aIndex, rew, preprocess(scr), term
      -- TODO: Save preprocessed transition safely (perhaps checking terminal?) to save computation

      -- Clamp reward for stability
      reward = math.min(reward, -opt.rewardClamp)
      reward = math.max(reward, opt.rewardClamp)

      -- Store in memory
      memory.store(state, action, reward, transition, terminal)
      state, action, reward, transition, terminal = nil, nil, nil, nil, nil -- TODO: Sanity check to remove later

      -- Occasionally sample from from memory
      if opt.step % opt.memSampleFreq == 0 and memory.size() >= opt.batchSize then
        -- Sample experience uniformly
        local indices = torch.randperm(memory.size()):long()
        indices = indices[{{1, opt.batchSize}}]
        -- Optimise (learn) from experience tuples
        self:optimise(memory.retrieve(indices))
      end
    end

    return scr, rew, term
  end

  -- Learns from experience (in batches)
  local learn = function(states, actions, rewards, transitions, terminals)
    -- Calculate max Q-value from next state
    local QMax = torch.max(net:forward(transitions), 2)
    -- Calculate target Y
    local Y = torch.add(rewards, torch.mul(QMax, opt.gamma)) -- TODO: Add target network and Double Q calculation
    -- Set target Y to reward if the transition was terminal
    Y[terminals] = rewards[terminals]

    -- Get all predicted Q-values from the current state
    local QCurr = net:forward(states)
    local QTaken = torch.CudaTensor(opt.batchSize)
    -- Get prediction of current Q-values with given actions
    for q = 1, opt.batchSize do
      QTaken[q] = QCurr[q][actions[q]]
    end

    -- Calculate TD errors δ
    local tdErr = QTaken - Y
    -- Clamp TD errors δ (approximates Huber loss)
    tdErr:clamp(-opt.tdClamp, opt.tdClamp)
    
    -- Zero QCurr outputs (no error)
    QCurr:zero()
    -- Set TD errors δ with given actions
    for q = 1, opt.batchSize do
      QCurr[q][actions[q]] = tdErr[q]
    end

    -- Reset gradients dθ
    dTheta:zero()
    -- Backpropagate gradients (network modifies internally)
    net:backward(states, QCurr)
    
    -- Calculate squared error loss (for optimiser)
    local loss = torch.mean(tdErr:pow(2))

    return loss, dTheta
  end

  -- "Optimises" the model parameters θ
  agent.optimise = function(self, states, actions, rewards, transitions, terminals)
    -- Create function to evaluate given parameters x
    local feval = function(x)
      return learn(states, actions, rewards, transitions, terminals)
    end
    
    local __, loss = optim[opt.optimiser](feval, theta, optimParams)
    return loss[1]
  end

  return agent
end

return model
