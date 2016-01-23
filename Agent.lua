local _ = require 'moses'
local class = require 'classic'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local Model = require 'Model'
local Experience = require 'Experience'
local CircularQueue = require 'structures/CircularQueue'
require 'classic.torch' -- Enables serialisation
require 'modules/rmspropm' -- Add RMSProp with momentum

local Agent = classic.class('Agent')

-- Creates a DQN agent
function Agent:_init(gameEnv, opt)
  -- Experiment ID
  self._id = opt._id
  -- Actions
  self.actionSpec = gameEnv:getActionSpec()
  self.m = self.actionSpec[3][2] -- Number of discrete actions

  -- Initialise model helper
  self.model = Model(opt)
  -- Create policy and target networks
  self.policyNet = self.model:create(self.m)
  self.targetNet = self.policyNet:clone() -- Create deep copy for target network
  self.tau = opt.tau
  self.doubleQ = opt.doubleQ
  -- Network parameters θ and gradients dθ
  self.theta, self.dTheta = self.policyNet:getParameters()

  -- Reinforcement learning parameters
  self.gamma = opt.gamma
  self.rewardClip = opt.rewardClip
  self.tdClip = opt.tdClip
  self.epsilonStart = opt.epsilonStart
  self.epsilonEnd = opt.epsilonEnd
  self.epsilonGrad = (opt.epsilonEnd - opt.epsilonStart)/opt.epsilonSteps -- Greediness ε decay factor
  self.PALpha = opt.PALpha

  -- State buffer
  self.stateBuffer = CircularQueue(opt.histLen, opt.Tensor, {opt.nChannels, opt.height, opt.width})
  -- Experience replay memory
  self.memory = Experience(opt.memSize, opt)
  self.memSampleFreq = opt.memSampleFreq
  self.memNReplay = opt.memNReplay
  self.memPriority = opt.memPriority

  -- Training mode
  self.isTraining = false
  self.batchSize = opt.batchSize
  self.learnStart = opt.learnStart
  self.progFreq = opt.progFreq
  -- Optimiser parameters
  self.optimiser = opt.optimiser
  self.optimParams = {
    learningRate = opt.eta,
    momentum = opt.momentum
  }

  -- Q-learning variables
  self.APrimeMax = opt.Tensor(opt.batchSize, 1)
  self.APrimeMaxInds = opt.Tensor(opt.batchSize, 1)
  self.QPrimes = opt.Tensor(opt.batchSize, self.m)
  self.Y = opt.Tensor(opt.batchSize)
  self.QCurr = opt.Tensor(opt.batchSize, self.m)
  self.QTaken = opt.Tensor(opt.batchSize)
  self.tdErr = opt.Tensor(opt.batchSize)
  self.Qs = opt.Tensor(opt.batchSize, self.m)
  self.Q = opt.Tensor(opt.batchSize)
  self.V = opt.Tensor(opt.batchSize, 1)
  self.tdErrAL = opt.Tensor(opt.batchSize)
  self.QPrime = opt.Tensor(opt.batchSize)
  self.VPrime = opt.Tensor(opt.batchSize, 1)

  -- Validation variables
  self.valSize = opt.valSize
  self.valMemory = Experience(opt.valSize, opt) -- Validation experience replay memory
  self.losses = {}
  self.avgV = {} -- Running average of V
  self.avgTdErr = {} -- Running average of TD-error δ

  self.Tensor = opt.Tensor
  -- Keep reference to opt for opt.step
  self.opt = opt -- TODO: Keep internal step counter
end

-- Sets training mode
function Agent:training()
  self.isTraining = true
  self.stateBuffer:clear() -- Clears state buffer
end

-- Sets evaluation mode
function Agent:evaluate()
  self.isTraining = false
  self.stateBuffer:clear() -- Clears state buffer
end
  
-- Observes the results of the previous transition and chooses the next action to perform
function Agent:observe(reward, observation, terminal)
  -- Clip reward for stability
  reward = math.min(reward, -self.rewardClip)
  reward = math.max(reward, self.rewardClip)

  -- Process observation of current state
  observation = self.model:preprocess(observation)

  -- Store in buffer depending on terminal status
  if terminal then
    self.stateBuffer:pushReset(observation) -- Will clear buffer on next push
  else
    self.stateBuffer:push(observation)
  end
  -- Retrieve current and historical states from state buffer
  local state = self.stateBuffer:readAll()

  -- Set ε based on training vs. evaluation mode
  local epsilon = 0.001
  if self.isTraining then
    -- Use annealing ε
    epsilon = math.max(self.epsilonStart + (self.opt.step - 1)*self.epsilonGrad, self.epsilonEnd)
  end

  -- Choose action by ε-greedy exploration
  local aIndex = 1 -- In a terminal state, choose no-op by default
  if not terminal then
    if math.random() < epsilon then 
      aIndex = torch.random(1, self.m)
    else
      -- Choose best action
      local __, ind = torch.max(self.policyNet:forward(state), 1)
      aIndex = ind[1]
    end
  end

  -- If training
  if self.isTraining then
    -- Store experience tuple parts (including pre-emptive action)
    self.memory:store(reward, observation, terminal, aIndex)

    -- Collect validation transitions at the start
    if self.opt.step <= self.valSize then
      self.valMemory:store(reward, observation, terminal, aIndex)
    end

    -- Sample uniformly or with prioritised sampling
    if self.opt.step % self.memSampleFreq == 0 and self.opt.step >= self.learnStart then -- Assumes learnStart is greater than batchSize
      for n = 1, self.memNReplay do
        -- Optimise (learn) from experience tuples
        self:optimise(self.memory:sample(self.memPriority))
      end
    end

    -- Update target network every τ steps
    if self.opt.step % self.tau == 0 and self.opt.step >= self.learnStart then
      self.targetNet = self.policyNet:clone()
    end
  end

  return aIndex
end

-- Learns from experience
function Agent:learn(x, indices, ISWeights)
  -- Copy x to parameters θ if necessary
  if x ~= self.theta then
    self.theta:copy(x)
  end
  -- Reset gradients dθ
  self.dTheta:zero()

  -- Retrieve experience tuples
  local states, actions, rewards, transitions, terminals = self.memory:retrieve(indices)

  -- Calculate Q-values from transition using policy network
  self.QPrimes = self.policyNet:forward(transitions) -- Evaluate Q-values of argmax actions using policy network
  -- Perform argmax action selection on transition using policy network: argmax_a[Q(s', a; θpolicy)]
  self.APrimeMax, self.APrimeMaxInds = torch.max(self.QPrimes, 2)

  -- Double Q-learning: Q(s', argmax_a[Q(s', a; θpolicy)]; θtarget)
  if self.doubleQ then
    -- Calculate Q-values from transition using target network
    self.QPrimes = self.targetNet:forward(transitions) -- Evaluate Q-values of argmax actions using target network
  end

  -- Initially set target Y = Q(s', argmax_a[Q(s', a; θpolicy)]; θ), where final θ is either θpolicy (DQN) or θtarget (DDQN)
  for n = 1, self.batchSize do
    self.Y[n] = self.QPrimes[n][self.APrimeMaxInds[n][1]]
  end
  -- Calculate target Y := r + γ.Q(s', argmax_a[Q(s', a; θpolicy)]; θ)
  self.Y:mul(self.gamma):add(rewards)
  -- Set target Y := r if the transition was terminal as V(terminal) = 0
  self.Y[terminals] = rewards[terminals] -- Little use optimising over batch processing if terminal states are rare

  -- Get all predicted Q-values from the current state
  self.QCurr = self.policyNet:forward(states) -- Correct internal state of policy network before backprop
  -- Get prediction of current Q-values with given actions
  for n = 1, self.batchSize do
    self.QTaken[n] = self.QCurr[n][actions[n]]
  end

  -- Calculate TD-errors δ := ∆Q(s, a) = Y − Q(s, a)
  self.tdErr = self.Y - self.QTaken

  -- Calculate Advantage Learning update(s)
  if self.PALpha > 0 then
    -- Calculate Q(s, a) and V(s) using target network
    self.Qs = self.targetNet:forward(states)
    for n = 1, self.batchSize do
      self.Q[n] = self.Qs[n][actions[n]]
    end
    self.V = torch.max(self.Qs, 2)

    -- Calculate Advantage Learning update ∆ALQ(s, a) := δ − αPAL(V(s) − Q(s, a))
    self.tdErrAL = self.tdErr - self.V:add(-self.Q):mul(self.PALpha) -- TODO: Torch.CudaTensor:csub is missing

    -- Calculate Q(s', a) and V(s') using target network
    if not self.doubleQ then
      self.QPrimes = self.targetNet:forward(transitions) -- Evaluate Q-values of argmax actions using target network
    end
    for n = 1, self.batchSize do
      self.QPrime[n] = self.QPrimes[n][actions[n]]
    end
    self.VPrime = torch.max(self.QPrimes, 2)
    -- Set values to 0 for terminal states
    self.QPrime[terminals] = 0
    self.VPrime[terminals] = 0

    -- Calculate Persistent Advantage Learning update ∆PALQ(s, a) := max[∆ALQ(s, a), δ − αPAL(V(s') − Q(s', a))]
    self.tdErr = torch.max(torch.cat(self.tdErrAL, self.tdErr:add(-(self.VPrime:add(-self.QPrime):mul(self.PALpha))), 2), 2):squeeze() -- tdErrPAL TODO: Torch.CudaTensor:csub is missing
  end

  -- Calculate loss
  local loss
  if self.tdClip > 0 then
    -- Squared loss is used within clipping range, absolute loss is used outside (approximates Huber loss)
    local sqLoss = torch.cmin(torch.abs(self.tdErr), self.tdClip)
    local absLoss = torch.abs(self.tdErr) - sqLoss
    loss = torch.mean(sqLoss:pow(2):mul(0.5):add(absLoss:mul(self.tdClip)))

    -- Clip TD-errors δ
    self.tdErr:clamp(-self.tdClip, self.tdClip)
  else
    -- Squared loss
    loss = torch.mean(self.tdErr:clone():pow(2):mul(0.5))
  end
  -- Send TD-errors δ to be used as priorities
  self.memory:updatePriorities(indices, self.tdErr)
  
  -- Zero QCurr outputs (no error)
  self.QCurr:zero()
  -- Set TD-errors δ with given actions
  for n = 1, self.batchSize do
     -- Correct prioritisation bias with importance-sampling weights
    self.QCurr[n][actions[n]] = ISWeights[n] * self.tdErr[n]
  end

  -- Backpropagate (network modifies gradients internally)
  self.policyNet:backward(states, self.QCurr)
  -- Divide gradient by batch size
  self.dTheta:div(self.batchSize)
  -- Clip the norm of the gradients
  self.policyNet:gradParamClip(10)

  return loss, self.dTheta
end

-- Optimises the network parameters θ
function Agent:optimise(indices, ISWeights)
  -- Create function to evaluate given parameters x
  local feval = function(x)
    return self:learn(x, indices, ISWeights)
  end
  
  -- Optimise
  local __, loss = optim[self.optimiser](feval, self.theta, self.optimParams)
  -- Store loss
  if self.opt.step % self.progFreq == 0 then
    self.losses[#self.losses + 1] = loss[1]
  end

  return loss[1]
end

-- Reports stats for validation
function Agent:report()
  -- Validation variables
  local totalV, totalTdErr = 0, 0

  -- Loop over validation transitions
  local nBatches = math.ceil(self.valSize / self.batchSize)
  local ISWeights = self.Tensor(self.batchSize):fill(1)
  local startIndex, endIndex, batchSize, indices
  for n = 1, nBatches do
    startIndex = (n - 1)*self.batchSize + 1
    endIndex = n*self.batchSize
    batchSize = endIndex - startIndex + 1
    indices = torch.linspace(startIndex, endIndex, batchSize):long()

    -- Perform "learning" (without optimisation)
    self:learn(self.theta, indices, ISWeights:narrow(1, 1, batchSize))

    -- Calculate V and TD-error δ
    if self.PALpha == 0 then
      self.VPrime = torch.max(self.QPrimes, 2)
    end
    totalV = totalV + torch.sum(self.VPrime)
    totalTdErr = totalTdErr + torch.abs(self.tdErr):sum()
  end

  -- Average and insert values
  self.avgV[#self.avgV + 1] = totalV / self.valSize
  self.avgTdErr[#self.avgTdErr + 1] = totalTdErr / self.valSize

  -- Plot losses
  gnuplot.pngfigure(paths.concat('experiments', self._id, 'losses.png'))
  gnuplot.plot(torch.Tensor(self.losses))
  gnuplot.plotflush()
  -- Plot V
  gnuplot.pngfigure(paths.concat('experiments', self._id, 'Vs.png'))
  gnuplot.plot(torch.Tensor(self.avgV))
  gnuplot.plotflush()
  -- Plot TD-error δ
  gnuplot.pngfigure(paths.concat('experiments', self._id, 'TDErrors.png'))
  gnuplot.plot(torch.Tensor(self.avgTdErr))
  gnuplot.plotflush()

  return self.avgV[#self.avgV], self.avgTdErr[#self.avgTdErr]
end

-- Saves the network parameters θ
function Agent:saveWeights(path)
  torch.save(path, self.theta)
end

-- Loads network parameters θ
function Agent:loadWeights(path)
  self.theta = torch.load(path)
  self.targetNew = self.policyNet:clone()
end

return Agent
