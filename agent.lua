local _ = require 'moses'
require 'dpnn' -- for :gradParamClip()
local optim = require 'optim'
local model = require 'model'
local Experience = require 'Experience'
local CircularQueue = require 'structures/CircularQueue'

local agent = {}

-- Creates a DQN agent
agent.create = function(gameEnv, opt)
  local DQN = {}
  model.init(opt) -- Initialise model helper

  -- Actions
  local A = gameEnv:getActions()
  local m = _.size(A)

  -- Create buffers
  agent.buffers = {
    -- Processed screen and historical state
    state = torch.FloatTensor(opt.nChannels, opt.height, opt.width),
    history = opt.Tensor(opt.histLen, opt.nChannels, opt.height, opt.width),
    -- Experience tuples
    states = opt.Tensor(opt.batchSize, opt.histLen, opt.nChannels, opt.height, opt.width),
    actions = torch.ByteTensor(opt.batchSize),
    rewards = opt.Tensor(opt.batchSize),
    transitions = opt.Tensor(opt.batchSize, opt.histLen, opt.nChannels, opt.height, opt.width),
    terminals = torch.ByteTensor(opt.batchSize),
    -- Q-learning variables
    APrimeMax = opt.Tensor(opt.batchSize, 1),
    APrimeMaxInds = opt.Tensor(opt.batchSize, 1),
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

  -- Policy and target networks
  DQN.policyNet = model.create(m)
  DQN.targetNet = DQN.policyNet:clone() -- Create deep copy for target network
  -- Network parameters θ and gradients dθ
  local theta, dTheta = DQN.policyNet:getParameters()

  -- Experience replay memory
  DQN.memory = Experience(opt)
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
  
  -- Observes the results of the previous transition and chooses the next action to perform
  function DQN:observe(reward, observation, terminal)
    -- Clip reward for stability
    reward = math.min(reward, -opt.rewardClip)
    reward = math.max(reward, opt.rewardClip)

    -- Process observation of current state
    agent.buffers.state = model.preprocess(observation:select(1, 1))

    -- Store in buffer depending on terminal status
    if terminal then
      self.stateBuffer:pushReset(agent.buffers.state) -- Will clear buffer on next push
    else
      self.stateBuffer:push(agent.buffers.state)
    end
    -- Retrieve current and historical states from state buffer
    agent.buffers.history = self.stateBuffer:readAll()

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
        local __, ind = torch.max(self.policyNet:forward(agent.buffers.history), 1)
        aIndex = ind[1]
      end
    end

    -- If training
    if self.isTraining then
      -- Store experience tuple parts (including pre-emptive action)
      self.memory:store(reward, agent.buffers.state, terminal, aIndex)

      -- Sample uniformly or with prioritised sampling
      if opt.step % opt.memSampleFreq == 0 and opt.step >= opt.learnStart then -- Assumes learnStart is greater than batchSize
        for n = 1, opt.memNReplay do
          -- Optimise (learn) from experience tuples
          self:optimise(self.memory:sample(opt.memPriority))
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

  -- Learns from experience
  function DQN:learn(x, indices, ISWeights)
    -- Copy x to parameters θ if necessary
    if x ~= theta then
      theta:copy(x)
    end
    -- Reset gradients dθ
    dTheta:zero()

    -- Retrieve experience tuples
    agent.buffers.states, agent.buffers.actions, agent.buffers.rewards, agent.buffers.transitions, agent.buffers.terminals = self.memory:retrieve(indices)

    -- Perform argmax action selection on transition using policy network: argmax_a[Q(s', a; θpolicy)]
    agent.buffers.APrimeMax, agent.buffers.APrimeMaxInds = torch.max(self.policyNet:forward(agent.buffers.transitions), 2)
    -- Double Q-learning: Q(s', argmax_a[Q(s', a; θpolicy)]; θtarget)
    if opt.doubleQ then
      -- Calculate Q-values from transition using target network
      agent.buffers.QPrimes = self.targetNet:forward(agent.buffers.transitions) -- Evaluate Q-values of argmax actions using target network
    else
      -- Calculate Q-values from transition using policy network
      agent.buffers.QPrimes = self.policyNet:forward(agent.buffers.transitions) -- Evaluate Q-values of argmax actions using policy network
    end
    -- Similar to evaluation of V(s) for δ := Y − V(s) now, will be updated to Y later
    for q = 1, opt.batchSize do
      agent.buffers.Y[q] = agent.buffers.QPrimes[q][agent.buffers.APrimeMaxInds[q][1]]
    end    
    -- Calculate target Y := r + γ.Q(s', argmax_a[Q(s', a; θpolicy)]; θtarget) in DDQN case
    agent.buffers.Y:mul(opt.gamma):add(agent.buffers.rewards)
    -- Set target Y to reward if the transition was terminal as V(terminal) = 0
    agent.buffers.Y[agent.buffers.terminals] = agent.buffers.rewards[agent.buffers.terminals] -- Little use optimising over batch processing if terminal states are rare

    -- Get all predicted Q-values from the current state
    agent.buffers.QCurr = self.policyNet:forward(agent.buffers.states)
    -- Get prediction of current Q-values with given actions
    for q = 1, opt.batchSize do
      agent.buffers.QTaken[q] = agent.buffers.QCurr[q][agent.buffers.actions[q]]
    end

    -- Calculate TD-errors δ := ∆Q(s, a) = Y − Q(s, a)
    agent.buffers.tdErr = agent.buffers.Y - agent.buffers.QTaken

    -- Calculate advantage learning update
    if opt.PALpha > 0 then
      -- Calculate Q(s, a) and V(s) using target network
      agent.buffers.Qs = self.targetNet:forward(agent.buffers.states)
      for q = 1, opt.batchSize do
        agent.buffers.Q[q] = agent.buffers.Qs[q][agent.buffers.actions[q]]
      end
      agent.buffers.V = torch.max(agent.buffers.Qs, 2)
      -- Calculate Advantage Learning update ∆ALQ(s, a) := δ − αPAL(V(s) − Q(s, a))
      agent.buffers.tdErrAL = agent.buffers.tdErr - agent.buffers.V:add(-agent.buffers.Q):mul(opt.PALpha) -- TODO: Torch.CudaTensor:csub is missing
      -- Calculate Q(s', a) and V(s') using target network
      if not opt.doubleQ then
        agent.buffers.QPrimes = self.targetNet:forward(agent.buffers.transitions) -- Evaluate Q-values of argmax actions using target network
      end
      for q = 1, opt.batchSize do
        agent.buffers.QPrime[q] = agent.buffers.QPrimes[q][agent.buffers.actions[q]]
      end
      agent.buffers.VPrime = torch.max(agent.buffers.QPrimes, 2)
      -- Set values to 0 for terminal states
      agent.buffers.QPrime[agent.buffers.terminals] = 0
      agent.buffers.VPrime[agent.buffers.terminals] = 0
      -- Calculate Persistent Advantage learning update ∆PALQ(s, a) := max[∆ALQ(s, a), δ − αPAL(V(s') − Q(s', a))]
      agent.buffers.tdErr = torch.max(torch.cat(agent.buffers.tdErrAL, agent.buffers.tdErr:add(-(agent.buffers.VPrime:add(-agent.buffers.QPrime):mul(opt.PALpha))), 2), 2):squeeze() -- tdErrPAL TODO: Torch.CudaTensor:csub is missing
    end

    -- Clip TD-errors δ (approximates Huber loss)
    agent.buffers.tdErr:clamp(-opt.tdClip, opt.tdClip)
    -- Send TD-errors δ to be used as priorities
    self.memory:updatePriorities(indices, agent.buffers.tdErr)
    
    -- Zero QCurr outputs (no error)
    agent.buffers.QCurr:zero()
    -- Set TD-errors δ with given actions
    for q = 1, opt.batchSize do
      agent.buffers.QCurr[q][agent.buffers.actions[q]] = ISWeights[q] * agent.buffers.tdErr[q] -- Correct prioritisation bias with importance-sampling weights
    end

    -- Backpropagate gradients (network modifies internally)
    self.policyNet:backward(agent.buffers.states, agent.buffers.QCurr)
    -- Clip the norm of the gradients
    self.policyNet:gradParamClip(10)
    
    -- Calculate squared error loss (for optimiser)
    local loss = torch.mean(agent.buffers.tdErr:pow(2))

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
    torch.save(paths.concat(path, 'memory.t7'), self.memory)
  end

  -- Loads saved network parameters θ
  function DQN:load(path)
    theta = torch.load(paths.concat(path, 'DQN.t7'))
    self.targetNet = self.policyNet:clone()
    self.memory = torch.load(paths.concat(path, 'memory.t7'))
  end

  return DQN
end

return agent
