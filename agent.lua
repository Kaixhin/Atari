local _ = require 'moses'
require 'dpnn' -- for :gradParamClip()
local optim = require 'optim'
local model = require 'model'
local experience = require 'experience'
local CircularQueue = require 'structures/CircularQueue'

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
  -- State buffer
  DQN.stateBuffer = CircularQueue(opt.histLen, opt.Tensor, {opt.nChannels, opt.height, opt.width})
  -- Training mode
  DQN.isTraining = false
  -- Optimiser parameters
  local optimParams = {
    learningRate = opt.eta, -- TODO: Learning rate annealing superseded by β annealing?
    momentum = opt.momentum
  }
  -- Just for RMSprop, the default optimiser in DQN papers, set its momentum variable
  if opt.optimiser == 'rmsprop' then
    optimParams.alpha = opt.momentum
  end

  -- Sets training mode
  function DQN:training()
    self.isTraining = true
    self.stateBuffer:clear() -- Clears state buffer
  end

  -- Sets evaluation mode
  function DQN:evaluate()
    self.isTraining = false
    self.stateBuffer:clear() -- Clears state buffer
  end

  -- Preallocate memory for state and history
  local state = torch.FloatTensor(opt.nChannels, opt.height, opt.width)
  local history = opt.Tensor(opt.histLen, opt.nChannels, opt.height, opt.width)
  
  -- Observes the results of the previous transition and chooses the next action to perform
  function DQN:observe(reward, observation, terminal)
    -- Clip reward for stability
    reward = math.min(reward, -opt.rewardClip)
    reward = math.max(reward, opt.rewardClip)

    -- Process observation of current state
    model.preprocess(state, observation:select(1, 1), opt)
    -- Store in buffer depending on terminal status
    if terminal then
      self.stateBuffer:pushReset(state) -- Will clear buffer on next push
    else
      self.stateBuffer:push(state)
    end
    -- Retrieve current and historical states from state buffer
    self.stateBuffer:readAll(history)

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
        -- Choose best action
        local __, ind = torch.max(self.policyNet:forward(history), 1)
        aIndex = ind[1]
      end
    end

    -- If training
    if self.isTraining then
      -- Store experience tuple parts (including pre-emptive action)
      self.memory:store(reward, history, terminal, aIndex)

      -- Sample uniformly or with prioritised sampling
      if opt.step % opt.memSampleFreq == 0 and opt.step >= opt.learnStart then -- Assumes learnStart is greater than batchSize
        for n = 1, opt.memNReplay do
          -- Optimise (learn) from experience tuples
          self:optimise(self.memory:sample(opt.batchSize, opt.memPriority))
        end
      end

      -- Update target network every τ steps
      if opt.step % opt.tau == 0 and opt.step >= opt.learnStart then
        self.targetNet = self.policyNet:clone()
      end
    end

    -- Collect garbage manually to prevent running out of memory
    collectgarbage()

    return aIndex
  end

  -- Acts on the environment
  function DQN:act(aIndex)
    -- Perform step on environment
    return gameEnv:step(A[aIndex], self.isTraining)
  end

  -- Preallocate memory for experience tuples and learning variables
  local tuple = {
    states = opt.Tensor(opt.batchSize, opt.histLen, opt.nChannels, opt.height, opt.width),
    actions = torch.ByteTensor(opt.batchSize),
    rewards = opt.Tensor(opt.batchSize),
    transitions = opt.Tensor(opt.batchSize, opt.histLen, opt.nChannels, opt.height, opt.width),
    terminals = torch.ByteTensor(opt.batchSize)
  }
  local learn = {
    __ = opt.Tensor(opt.batchSize, 1),
    APrimeMax = opt.Tensor(opt.batchSize, 1),
    QPrimes = opt.Tensor(opt.batchSize, m),
    Y = opt.Tensor(opt.batchSize),
    QCurr = opt.Tensor(opt.batchSize, m), 
    QTaken = opt.Tensor(opt.batchSize),
    tdErr = opt.Tensor(opt.batchSize),
    Qs = opt.Tensor(opt.batchSize, m), 
    Q = opt.Tensor(opt.batchSize),
    V = opt.Tensor(opt.batchSize, 1),
    tdErrAL = opt.Tensor(opt.batchSize),
    QPrime = opt.Tensor(opt.batchSize),
    VPrime = opt.Tensor(opt.batchSize, 1)
  }

  -- Learns from experience
  function DQN:learn(x, indices, ISWeights)
    -- Copy x to parameters θ if necessary
    if x ~= theta then
      theta:copy(x)
    end
    -- Reset gradients dθ
    dTheta:zero()

    -- Retrieve experience tuples
    self.memory:retrieve(tuple, indices)

    -- Perform argmax action selection on transition using policy network: argmax_a[Q(s', a; θpolicy)]
    learn.__, learn.APrimeMax = torch.max(self.policyNet:forward(tuple.transitions), 2)
    -- Double Q-learning: Q(s', argmax_a[Q(s', a; θpolicy)]; θtarget)
    if opt.doubleQ then
      -- Calculate Q-values from transition using target network
      learn.QPrimes = self.targetNet:forward(tuple.transitions) -- Evaluate Q-values of argmax actions using target network
    else
      -- Calculate Q-values from transition using policy network
      learn.QPrimes = self.policyNet:forward(tuple.transitions) -- Evaluate Q-values of argmax actions using policy network
    end
    -- Similar to evaluation of V(s) for δ := Y − V(s) now, will be updated to Y later
    for q = 1, opt.batchSize do
      learn.Y[q] = learn.QPrimes[q][learn.APrimeMax[q][1]]
    end    
    -- Calculate target Y := r + γ.Q(s', argmax_a[Q(s', a; θpolicy)]; θtarget) in DDQN case
    learn.Y:mul(opt.gamma):add(tuple.rewards)
    -- Set target Y to reward if the transition was terminal as V(terminal) = 0
    learn.Y[tuple.terminals] = tuple.rewards[tuple.terminals] -- Little use optimising over batch processing if terminal states are rare

    -- Get all predicted Q-values from the current state
    learn.QCurr = self.policyNet:forward(tuple.states)
    -- Get prediction of current Q-values with given actions
    for q = 1, opt.batchSize do
      learn.QTaken[q] = learn.QCurr[q][tuple.actions[q]]
    end

    -- Calculate TD-errors δ := ∆Q(s, a) = Y − Q(s, a)
    learn.tdErr = learn.Y - learn.QTaken

    -- Calculate advantage learning update
    if opt.PALpha > 0 then
      -- Calculate Q(s, a) and V(s) using target network
      learn.Qs = self.targetNet:forward(tuple.states)
      for q = 1, opt.batchSize do
        learn.Q[q] = learn.Qs[q][tuple.actions[q]]
      end
      learn.V = torch.max(learn.Qs, 2)
      -- Calculate Advantage Learning update ∆ALQ(s, a) := δ − αPAL(V(s) − Q(s, a))
      learn.tdErrAL = learn.tdErr - learn.V:add(-learn.Q):mul(opt.PALpha) -- TODO: Torch.CudaTensor:csub is missing
      -- Calculate Q(s', a) and V(s') using target network
      if not opt.doubleQ then
        learn.QPrimes = self.targetNet:forward(tuple.transitions) -- Evaluate Q-values of argmax actions using target network
      end
      for q = 1, opt.batchSize do
        learn.QPrime[q] = learn.QPrimes[q][tuple.actions[q]]
      end
      learn.VPrime = torch.max(learn.QPrimes, 2)
      -- Set values to 0 for terminal states
      learn.QPrime[tuple.terminals] = 0
      learn.VPrime[tuple.terminals] = 0
      -- Calculate Persistent Advantage learning update ∆PALQ(s, a) := max[∆ALQ(s, a), δ − αPAL(V(s') − Q(s', a))]
      learn.tdErr = torch.max(torch.cat(learn.tdErrAL, learn.tdErr:add(-(learn.VPrime:add(-learn.QPrime):mul(opt.PALpha))), 2), 2):squeeze() -- tdErrPAL TODO: Torch.CudaTensor:csub is missing
    end

    -- Clip TD-errors δ (approximates Huber loss)
    learn.tdErr:clamp(-opt.tdClip, opt.tdClip)
    -- Send TD-errors δ to be used as priorities
    self.memory:updatePriorities(indices, learn.tdErr:clone())
    
    -- Zero QCurr outputs (no error)
    learn.QCurr:zero()
    -- Set TD-errors δ with given actions
    for q = 1, opt.batchSize do
      learn.QCurr[q][tuple.actions[q]] = ISWeights[q] * learn.tdErr[q] -- Correct prioritisation bias with importance-sampling weights
    end

    -- Backpropagate gradients (network modifies internally)
    self.policyNet:backward(tuple.states, learn.QCurr)
    -- Clip the norm of the gradients
    self.policyNet:gradParamClip(10)
    
    -- Calculate squared error loss (for optimiser)
    local loss = torch.mean(learn.tdErr:pow(2))

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

  -- Saves the network parameters θ
  function DQN:save(path)
    torch.save(paths.concat(path, 'DQN.t7'), theta)
  end

  -- Loads saved network parameters θ
  function DQN:load(path)
    theta = torch.load(path)
    self.targetNet = self.policyNet:clone()
  end

  return DQN
end

return agent
