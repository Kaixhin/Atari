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
  -- Experience replay memory (also retains history)
  DQN.memory = experience.create(opt)
  -- Training mode
  DQN.isTraining = false
  -- Optimiser parameters
  local optimParams = {
    learningRate = opt.eta, -- TODO: Learning rate annealing superseded by β annealing?
    momentum = opt.momentum
  }
  -- Just for RMSprop, the seeming default for DQNs, set its momentum variable
  if opt.optimiser == 'rmsprop' then
    optimParams.alpha = opt.momentum
  end

  -- Sets training mode
  function DQN:training()
    self.isTraining = true
  end

  -- Sets evaluation mode
  function DQN:evaluate()
    self.isTraining = false
  end
  
  -- Observes the results of the previous transition and chooses the next action to perform
  function DQN:observe(reward, observation, terminal)
    -- Clip reward for stability
    reward = math.min(reward, -opt.rewardClip)
    reward = math.max(reward, opt.rewardClip)

    -- Process observation of current state
    local state = model.preprocess(observation:select(1, 1), opt) 

    -- Set ε based on training vs. evaluation mode
    local epsilon = 0.001
    if self.isTraining then
      -- Use annealing ε
      epsilon = opt.epsilon[opt.step]
    end

    -- Choose action by ε-greedy exploration
    local aIndex = 1 -- In a terminal state, choose no-op by default
    if not terminal then
      if math.random() < epsilon then 
        aIndex = torch.random(1, m)
      else
        -- Retrieve current and historical states
        local history = self.memory:retrieveHistory()
        -- Choose best action
        local __, ind = torch.max(self.policyNet:forward(history), 1)
        aIndex = ind[1]
      end
    end

    -- Store experience tuple parts (including pre-emptive action)
    self.memory:store(reward, state, terminal, aIndex)

    -- If training
    if self.isTraining then
      -- Sample uniformly or with prioritised sampling
      if opt.step % opt.memSampleFreq == 0 and opt.step >= opt.learnStart then -- Assumes learnStart is greater than batchSize
        for n = 1, opt.memNReplay do
          -- Optimise (learn) from experience tuples
          self:optimise(self.memory:sample(opt.batchSize, opt.memPriority))
        end
      end

      -- Update target network every τ steps
      if opt.step % opt.tau == 0 and opt.step >= opt.learnStart then
        local targetTheta = self.targetNet:getParameters()
        targetTheta:copy(theta) -- Deep copy policy network parameters
      end
    end

    return aIndex
  end

  -- Acts on the environment
  function DQN:act(aIndex)
    -- Perform step on environment
    return gameEnv:step(A[aIndex], self.isTraining)
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

    -- Perform argmax action selection on transition using policy network: argmax_a[Q(s', a; θpolicy)]
    local __, APrimeMax = torch.max(self.policyNet:forward(transitions), 2)
    local QPrimes
    -- Double Q-learning: Q(s', argmax_a[Q(s', a; θpolicy)]; θtarget)
    if opt.doubleQ then
      -- Calculate Q-values from transition using target network
      QPrimes = self.targetNet:forward(transitions) -- Evaluate Q-values of argmax actions using target network
    else
      -- Calculate Q-values from transition using policy network
      QPrimes = self.policyNet:forward(transitions) -- Evaluate Q-values of argmax actions using policy network
    end
    local Y = opt.Tensor(opt.batchSize) -- Similar to evaluation of V(s) for δ := Y − V(s) now, will be updated to Y later
    for q = 1, opt.batchSize do
      Y[q] = QPrimes[q][APrimeMax[q][1]]
    end    
    -- Calculate target Y := r + γ.Q(s', argmax_a[Q(s', a; θpolicy)]; θtarget) in DDQN case
    Y:mul(opt.gamma):add(rewards)
    -- Set target Y to reward if the transition was terminal as V(terminal) = 0
    Y[terminals] = rewards[terminals] -- Little use optimising over batch processing if terminal states are rare

    -- Get all predicted Q-values from the current state
    local QCurr = self.policyNet:forward(states)
    local QTaken = opt.Tensor(opt.batchSize)
    -- Get prediction of current Q-values with given actions
    for q = 1, opt.batchSize do
      QTaken[q] = QCurr[q][actions[q]]
    end

    -- Calculate TD-errors δ := ∆Q(s, a) = Y − Q(s, a)
    local tdErr = Y - QTaken
    -- Calculate advantage learning update
    if opt.PALpha > 0 then
      -- Calculate Q(s, a) and V(s) using target network
      local Qs = self.targetNet:forward(states)
      local Q = opt.Tensor(opt.batchSize)
      for q = 1, opt.batchSize do
        Q[q] = Qs[q][actions[q]]
      end
      local V = torch.max(Qs, 2)
      -- Calculate Advantage Learning update ∆ALQ(s, a) := δ − αPAL(V(s) − Q(s, a))
      local tdErrAL = tdErr - V:add(-Q):mul(opt.PALpha) -- TODO: Torch.CudaTensor:csub is missing
      -- Calculate Q(s', a) and V(s') using target network
      if not opt.doubleQ then
        QPrimes = self.targetNet:forward(transitions) -- Evaluate Q-values of argmax actions using target network
      end
      local QPrime = opt.Tensor(opt.batchSize)
      for q = 1, opt.batchSize do
        QPrime[q] = QPrimes[q][actions[q]]
      end
      local VPrime = torch.max(QPrimes, 2)
      -- Set values to 0 for terminal states
      QPrime[terminals] = 0
      VPrime[terminals] = 0
      -- Calculate Persistent Advantage learning update ∆PALQ(s, a) := max[∆ALQ(s, a), δ − αPAL(V(s') − Q(s', a))]
      tdErr = torch.max(torch.cat(tdErrAL, tdErr:add(-(VPrime:add(-QPrime):mul(opt.PALpha))), 2), 2):squeeze() -- tdErrPAL TODO: Torch.CudaTensor:csub is missing
    end

    -- Clip TD-errors δ (approximates Huber loss)
    tdErr:clamp(-opt.tdClip, opt.tdClip)
    -- Send TD-errors δ to be used as priorities
    self.memory:updatePriorities(indices, tdErr)
    
    -- Zero QCurr outputs (no error)
    QCurr:zero()
    -- Set TD-errors δ with given actions
    for q = 1, opt.batchSize do
      QCurr[q][actions[q]] = ISWeights[q] * tdErr[q] -- Correct prioritisation bias with importance-sampling weights
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
    local targetTheta = self.targetNet:getParameters()
    targetTheta:copy(theta) -- Deep copy policy network parameters
  end

  return DQN
end

return agent
