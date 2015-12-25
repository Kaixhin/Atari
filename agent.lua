local _ = require 'moses'
require 'dpnn' -- for :gradParamClip()
local optim = require 'optim'
local model = require 'model'
local experience = require 'experience'

local agent = {}

-- Creates a DQN agent
agent.create = function(gameEnv, opt)
  local DQN = {}
  local A = gameEnv:getActions()
  local m = _.size(A)
  -- Actor and target networks
  DQN.net = model.create(A, opt)
  DQN.targetNet = DQN.net:clone() -- Create deep copy for target network
  -- Network parameters θ and gradients dθ
  local theta, dTheta = DQN.net:getParameters()
  -- Experience replay memory
  DQN.memory = experience.create(opt)
  -- Training mode
  DQN.isTraining = false
  -- Learning variable updated on observe()
  DQN.state = nil
  -- Learning variables updated on act()
  DQN.action, DQN.reward, DQN.transition, DQN.terminal = nil, nil, nil, nil
  -- Optimiser parameters
  local optimParams = {
    learningRate = opt.eta, -- TODO: Check learning rate annealing parameters
    alpha = opt.momentum -- TODO: Select proper momentum for optimisers other than RMSprop
  }

  -- Sets training mode
  DQN.training = function(self)
    self.isTraining = true
  end

  -- Sets evaluation mode
  DQN.evaluate = function(self)
    self.isTraining = false
    -- Reset learning variables
    self.state, self.action, self.reward, self.transition, self.terminal = nil, nil, nil, nil, nil
  end
  
  -- Outputs an action (index) to perform on the environment
  DQN.observe = function(self, observation)
    local state
    -- Use preprocessed transition if available
    if self.state then
      state = self.state
    else
      state = model.preprocess(observation, opt)
    end
    local aIndex

    -- Set ε based on training vs. evaluation mode
    local epsilon = 0.001
    if self.isTraining then
      -- Use annealing ε
      epsilon = opt.epsilon[opt.step] 
      
      -- Store state
      self.state = state
    end

    -- Choose action by ε-greedy exploration
    if math.random() < epsilon then 
      aIndex = torch.random(1, m)
    else
      local __, ind = torch.max(self.net:forward(state), 1)
      aIndex = ind[1]
    end

    return aIndex
  end

  -- Acts on (and can learn from) the environment
  DQN.act = function(self, aIndex)
    local screen, reward, terminal = gameEnv:step(A[aIndex], self.isTraining)

    -- If training
    if self.isTraining then
      -- Store action (index), reward, transition and terminal
      self.action, self.reward, self.transition, self.terminal = aIndex, reward, model.preprocess(screen, opt), terminal

      -- Clamp reward for stability
      self.reward = math.min(self.reward, -opt.rewardClamp)
      self.reward = math.max(self.reward, opt.rewardClamp)

      -- Store in memory
      self.memory:store(self.state, self.action, self.reward, self.transition, self.terminal)
      -- Store preprocessed transition as state for performance
      if not terminal then
        self.state = self.transition
      else
        self.state = nil
      end

      -- Occasionally sample from from memory
      if opt.step % opt.memSampleFreq == 0 and self.memory:size() >= opt.batchSize then
        -- Sample with proportional prioritised sampling
        local indices, ISWeights = self.memory:prioritySample(opt.batchSize)
        -- Optimise (learn) from experience tuples
        self:optimise(indices, ISWeights)
      end

      -- Update target network every τ steps
      if opt.step % opt.tau == 0 then
        local targetTheta = self.targetNet:getParameters()
        targetTheta:copy(theta) -- Deep copy network parameters
      end
    end

    return screen, reward, terminal
  end

  -- Learns from experience
  DQN.learn = function(self, indices, ISWeights)
    -- Retrieve experience tuples
    local states, actions, rewards, transitions, terminals = self.memory:retrieve(indices)

    -- Perform argmax action selection using network
    local __, AMax = torch.max(self.net:forward(transitions), 2)
    -- Calculate Q-values from next state using target network
    local QTargets = self.targetNet:forward(transitions)
    -- Evaluate Q-values of argmax actions using target network (Double Q-learning)
    local QMax = torch.Tensor(opt.batchSize)
    if opt.gpu > 0 then
      QMax = QMax:cuda()
    end
    for q = 1, opt.batchSize do
      QMax[q] = QTargets[q][AMax[q][1]]
    end    
    -- Calculate target Y
    local Y = torch.add(rewards, torch.mul(QMax, opt.gamma))
    -- Set target Y to reward if the transition was terminal
    Y[terminals] = rewards[terminals] -- Little use optimising over batch processing if terminal states are rare

    -- Get all predicted Q-values from the current state
    local QCurr = self.net:forward(states)
    local QTaken = torch.Tensor(opt.batchSize)
    if opt.gpu > 0 then
      QTaken = QTaken:cuda()
    end
    -- Get prediction of current Q-values with given actions
    for q = 1, opt.batchSize do
      QTaken[q] = QCurr[q][actions[q]]
    end

    -- Calculate TD-errors δ (to minimise; Y - QTaken would require gradient ascent)
    local tdErr = QTaken - Y
    -- Clamp TD-errors δ (approximates Huber loss)
    tdErr:clamp(-opt.tdClamp, opt.tdClamp)
    -- Store magnitude of TD-errors δ as priorities
    self.memory:updatePriorities(indices, torch.abs(tdErr))
    
    -- Zero QCurr outputs (no error)
    QCurr:zero()
    -- Set TD-errors δ with given actions
    for q = 1, opt.batchSize do
      QCurr[q][actions[q]] = ISWeights[q] * tdErr[q] -- Correct prioritisation bias with importance-sampling weights
    end

    -- Reset gradients dθ
    dTheta:zero()
    -- Backpropagate gradients (network modifies internally)
    self.net:backward(states, QCurr)
    -- Clip the norm of the gradients
    self.net:gradParamClip(10)
    
    -- Calculate squared error loss (for optimiser)
    local loss = torch.mean(tdErr:pow(2))

    return loss, dTheta
  end

  -- "Optimises" the network parameters θ
  DQN.optimise = function(self, indices, ISWeights)
    -- Create function to evaluate given parameters x
    local feval = function(x)
      return self:learn(indices, ISWeights)
    end
    
    local __, loss = optim[opt.optimiser](feval, theta, optimParams)
    return loss[1]
  end

  return DQN
end

return agent
