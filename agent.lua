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
  -- Policy and target networks
  DQN.policyNet = model.create(A, opt)
  DQN.targetNet = DQN.policyNet:clone() -- Create deep copy for target network
  -- Network parameters θ and gradients dθ
  local theta, dTheta = DQN.policyNet:getParameters()
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
    learningRate = opt.eta, -- TODO: Learning rate annealing superseded by β annealing?
    momentum = opt.momentum
  }
  if opt.optimiser == 'rmsprop' then
    -- Just for RMSprop, the seeming default for DQNs, set its momentum variable
    optimParams.alpha = opt.momentum
  end

  -- Sets training mode
  function DQN:training()
    self.isTraining = true
  end

  -- Sets evaluation mode
  function DQN:evaluate()
    self.isTraining = false
    -- Reset learning variables
    self.state, self.action, self.reward, self.transition, self.terminal = nil, nil, nil, nil, nil
  end
  
  -- Outputs an action (index) to perform on the environment
  function DQN:observe(observation)
    local state
    -- Use preprocessed transition if available
    state = self.state or model.preprocess(observation, opt)
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
      local __, ind = torch.max(self.policyNet:forward(state), 1)
      aIndex = ind[1]
    end

    return aIndex
  end

  -- Acts on (and can learn from) the environment
  function DQN:act(aIndex)
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
      if opt.step % opt.memSampleFreq == 0 and opt.step >= opt.learnStart then -- Assumes learnStart is greater than batchSize
        -- Sample uniformly or with prioritised sampling
        local indices, ISWeights = self.memory:prioritySample(opt.memPriority)
        -- Optimise (learn) from experience tuples
        self:optimise(indices, ISWeights)
      end

      -- Update target network every τ steps
      if opt.step % opt.tau == 0 and opt.step >= opt.learnStart then
        local targetTheta = self.targetNet:getParameters()
        targetTheta:copy(theta) -- Deep copy policy network parameters
      end
    end

    return screen, reward, terminal
  end

  -- Learns from experience
  function DQN:learn(x, indices, ISWeights)
    -- Copy x to parameters θ if necessary
    if x ~= theta then
      theta:copy(x)
    end
    -- Reset gradients dθ
    dTheta:zero()

    -- Retrieve experience tuples
    local states, actions, rewards, transitions, terminals = self.memory:retrieve(indices)

    -- Perform argmax action selection on transition using policy network
    local __, APrimeMax = torch.max(self.policyNet:forward(transitions), 2)
    -- Calculate Q-values from transition using target network
    local QPrimeTargets = self.targetNet:forward(transitions)
    -- Evaluate Q-values of argmax actions using target network (Double Q-learning)
    local QPrimeMax = opt.Tensor(opt.batchSize) -- Note that this is therefore the estimate of V(state)
    for q = 1, opt.batchSize do
      QPrimeMax[q] = QPrimeTargets[q][APrimeMax[q][1]]
    end    
    -- Calculate target Y
    local Y = torch.add(rewards, torch.mul(QPrimeMax, opt.gamma))
    -- Set target Y to reward if the transition was terminal
    Y[terminals] = rewards[terminals] -- Little use optimising over batch processing if terminal states are rare

    -- Get all predicted Q-values from the current state
    local QCurr = self.policyNet:forward(states)
    local QTaken = opt.Tensor(opt.batchSize)
    -- Get prediction of current Q-values with given actions
    for q = 1, opt.batchSize do
      QTaken[q] = QCurr[q][actions[q]]
    end

    -- Calculate TD-errors δ (to minimise)
    local tdErr = Y - QTaken
    -- Calculate Q(state, action) and V(state) using target network -- TODO: Check if Q(s, a) is TD-error or from target network
    local Qs = self.targetNet:forward(states) -- Calculate all Q-values
    local Q = opt.Tensor(opt.batchSize)
    for q = 1, opt.batchSize do
      Q[q] = Qs[q][actions[q]]
    end
    local V = torch.max(Qs, 2)
    -- Calculate Advantage Learning update
    local tdErrAL = tdErr - torch.mul(torch.add(V, -Q), opt.PALpha)
    -- Calculate Q(transition, action) and V(transition) using target network
    local QPrimes = self.targetNet:forward(transitions) -- Calculate all Q-values
    local QPrime = opt.Tensor(opt.batchSize)
    for q = 1, opt.batchSize do
      QPrime[q] = QPrimes[q][actions[q]]
    end
    local VPrime = torch.max(QPrimes, 2)
    -- Calculate Persistent Advantage learning update
    local tdErrPAL = torch.max(torch.cat(tdErrAL, tdErr - torch.mul(torch.add(VPrime, -QPrime), opt.PALpha), 2), 2):squeeze()

    -- Clamp TD-errors δ (approximates Huber loss)
    tdErrPAL:clamp(-opt.tdClamp, opt.tdClamp)
    -- Send TD-errors δ to be used as priorities
    self.memory:updatePriorities(indices, tdErrPAL)
    
    -- Zero QCurr outputs (no error)
    QCurr:zero()
    -- Set TD-errors δ with given actions
    for q = 1, opt.batchSize do
      QCurr[q][actions[q]] = ISWeights[q] * tdErrPAL[q] -- Correct prioritisation bias with importance-sampling weights
    end

    -- Backpropagate gradients (network modifies internally)
    self.policyNet:backward(states, QCurr)
    -- Clip the norm of the gradients
    self.policyNet:gradParamClip(10)
    
    -- Calculate squared error loss (for optimiser)
    local loss = torch.mean(tdErr:pow(2))

    return loss, dTheta
  end

  -- "Optimises" the network parameters θ
  function DQN:optimise(indices, ISWeights)
    -- Create function to evaluate given parameters x
    local feval = function(x)
      return self:learn(x, indices, ISWeights)
    end
    
    local __, loss = optim[opt.optimiser](feval, theta, optimParams)
    return loss[1]
  end

  -- Saves the network
  function DQN:save(path)
    torch.save(paths.concat(path, 'DQN.t7'), theta)
  end

  -- Loads a saved network
  function DQN:load(path)
    theta = torch.load(path)
  end

  return DQN
end

return agent
