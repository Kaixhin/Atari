local _ = require 'moses'
require 'cutorch'
local optim = require 'optim'
local model = require 'model'
local experience = require 'experience'

local agent = {}

-- Creates a DQN agent
agent.create = function(gameEnv, opt)
  local DQN = {}
  local A = gameEnv:getActions()
  -- DQN
  local net = model.create(A)
  local targetNet = net:clone() -- Create deep copy for target network
  -- Network parameters θ and gradients dθ
  local theta, dTheta = net:getParameters()
  -- Experience replay memory
  local memory = experience.createMemory(opt.memSize, {1, 84, 84})
  -- Training mode
  DQN.isTraining = false
  -- Experience variables for learning
  local state -- Updated on observe
  local action, reward, transition, terminal -- Updated on act
  -- Optimiser parameters
  local optimParams = {
    learningRate = opt.alpha, -- TODO: Check learning rate annealing parameters
    alpha = opt.momentum -- TODO: Select proper momentum for optimisers other than RMSprop
  }

  -- Sets training mode
  DQN.training = function(self)
    self.isTraining = true
  end

  -- Sets evaluation mode
  DQN.evaluate = function(self)
    self.isTraining = false
  end
  
  -- Outputs an action (index) to perform on the environment
  DQN.observe = function(self, observation)
    local s
    -- Use preprocessed transition if available
    if state then
      s = state
    else
      s = model.preprocess(observation)
    end
    local aIndex

    -- Set ε based on training vs. evaluation mode
    local epsilon = 0.001
    if self.isTraining then
      -- Use annealing ε
      epsilon = opt.epsilon[opt.step] 
      
      -- Store state
      state = s
    end

    -- Choose action by ε-greedy exploration
    if math.random() < epsilon then 
      aIndex = torch.random(1, _.size(A))
    else
      local __, ind = torch.max(net:forward(s), 1)
      aIndex = ind[1]
    end

    return aIndex
  end

  -- Acts on the environment (and can learn)
  DQN.act = function(self, aIndex)
    local scr, rew, term = gameEnv:step(A[aIndex], self.isTraining)

    -- If training
    if self.isTraining then
      -- Store action (index), reward, transition and terminal
      action, reward, transition, terminal = aIndex, rew, model.preprocess(scr), term

      -- Clamp reward for stability
      reward = math.min(reward, -opt.rewardClamp)
      reward = math.max(reward, opt.rewardClamp)

      -- Store in memory
      memory.store(state, action, reward, transition, terminal)
      -- Store preprocessed transition as state for performance
      if not terminal then
        state = transition
      else
        state = nil
      end

      -- Occasionally sample from from memory
      if opt.step % opt.memSampleFreq == 0 and memory.size() >= opt.batchSize then
        -- Sample experience uniformly
        local indices = torch.randperm(memory.size()):long()
        indices = indices[{{1, opt.batchSize}}]
        -- Optimise (learn) from experience tuples
        self:optimise(memory.retrieve(indices))
      end

      -- Update target network every τ steps
      if opt.step % opt.tau == 0 then
        local targetTheta = targetNet:getParameters()
        targetTheta:copy(theta) -- Deep copy network parameters
      end
    end

    return scr, rew, term
  end

  -- Learns from experience (in batches)
  local learn = function(states, actions, rewards, transitions, terminals)
    -- Perform argmax action selection using network
    local __, AMax = torch.max(net:forward(transitions), 2)
    -- Calculate Q-values from next state using target network
    local QTargets = targetNet:forward(transitions)
    -- Evaluate Q-values of argmax actions using target network (Double Q-learning)
    local QMax = torch.CudaTensor(opt.batchSize)
    for q = 1, opt.batchSize do
      QMax[q] = QTargets[q][AMax[q][1]]
    end    
    -- Calculate target Y
    local Y = torch.add(rewards, torch.mul(QMax, opt.gamma))
    -- Set target Y to reward if the transition was terminal
    Y[terminals] = rewards[terminals] -- Little use optimising over batch processing if terminal states are rare

    -- Get all predicted Q-values from the current state
    local QCurr = net:forward(states)
    local QTaken = torch.CudaTensor(opt.batchSize)
    -- Get prediction of current Q-values with given actions
    for q = 1, opt.batchSize do
      QTaken[q] = QCurr[q][actions[q]]
    end

    -- Calculate TD errors δ (to minimise; Y - QTaken would require gradient ascent)
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
    -- Clip the norm of the gradients
    net:gradParamClip(10)
    
    -- Calculate squared error loss (for optimiser)
    local loss = torch.mean(tdErr:pow(2))

    return loss, dTheta
  end

  -- "Optimises" the network parameters θ
  DQN.optimise = function(self, states, actions, rewards, transitions, terminals)
    -- Create function to evaluate given parameters x
    local feval = function(x)
      return learn(states, actions, rewards, transitions, terminals)
    end
    
    local __, loss = optim[opt.optimiser](feval, theta, optimParams)
    return loss[1]
  end

  return DQN
end

return agent
