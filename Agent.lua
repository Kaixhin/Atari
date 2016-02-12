local _ = require 'moses'
local class = require 'classic'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local Model = require 'Model'
local Experience = require 'Experience'
local CircularQueue = require 'structures/CircularQueue'
local Singleton = require 'structures/Singleton'
require 'classic.torch' -- Enables serialisation
require 'modules/rmspropm' -- Add RMSProp with momentum

local Agent = classic.class('Agent')

-- Creates a DQN agent
function Agent:_init(env, opt)
  -- Experiment ID
  self._id = opt._id
  -- Actions
  self.actionSpec = env:getActionSpec()
  self.m = self.actionSpec[3][2] - self.actionSpec[3][1] + 1 -- Number of discrete actions
  self.actionOffset = 1 - self.actionSpec[3][1] -- Calculate offset if first action is not indexed as 1

  -- Initialise model helper
  self.model = Model(opt)
  -- Create policy and target networks
  self.policyNet = self.model:create(self.m)
  self.targetNet = self.policyNet:clone() -- Create deep copy for target network
  self.targetNet:evaluate() -- Target network always in evaluation mode
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
  self.memNSamples = opt.memNSamples
  self.memPriority = opt.memPriority

  -- Training mode
  self.isTraining = false
  self.batchSize = opt.batchSize
  self.learnStart = opt.learnStart
  self.progFreq = opt.progFreq
  self.gradClip = opt.gradClip
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
  self.avgV = {} -- Running average of V(s')
  self.avgTdErr = {} -- Running average of TD-error δ
  self.valScores = {} -- Validation scores (passed from main script)

  -- Tensor creation
  self.Tensor = opt.Tensor

  -- Saliency display
  self:setSaliency(opt.saliency) -- Set saliency option on agent and model
  self.origWidth = opt.origWidth
  self.origHeight = opt.origHeight
  self.saliencyMap = opt.Tensor(1, opt.origHeight, opt.origWidth)
  self.histLen = opt.histLen
  self.inputGrads = opt.Tensor(opt.histLen*opt.nChannels, opt.height, opt.width) -- Gradients with respect to the input (for saliency maps)

  -- Get singleton instance for step
  self.globals = Singleton.getInstance()
end

-- Sets training mode
function Agent:training()
  self.isTraining = true
  self.policyNet:training()
  -- Clear state buffer
  self.stateBuffer:clear()
end

-- Sets evaluation mode
function Agent:evaluate()
  self.isTraining = false
  self.policyNet:evaluate()
  -- Clear state buffer
  self.stateBuffer:clear()
  -- Set previously stored state as invalid (as no transition stored)
  self.memory:setInvalid()
end
  
-- Observes the results of the previous transition and chooses the next action to perform
function Agent:observe(reward, observation, terminal)
  -- Clip reward for stability
  if self.rewardClip > 0 then
    reward = math.min(reward, -self.rewardClip)
    reward = math.max(reward, self.rewardClip)
  end

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
    if self.globals.step < self.learnStart then
      -- Keep ε constant before learning starts
      epsilon = self.epsilonStart
    else
      -- Use annealing ε
      epsilon = math.max(self.epsilonStart + (self.globals.step - self.learnStart - 1)*self.epsilonGrad, self.epsilonEnd)
    end
  end

  -- Choose action by ε-greedy exploration
  local aIndex = 1 -- In a terminal state, choose no-op/first action by default
  if not terminal then
    if math.random() < epsilon then 
      aIndex = torch.random(1, self.m)

      -- Reset saliency if action not chosen by network
      if self.saliency ~= 'none' then
        self.saliencyMap:zero()
      end
    else
      -- Evaluate state
      local Qs = self.policyNet:forward(state)
      local maxQ = Qs[1]
      local bestAs = {1}
      -- Find best actions
      for a = 2, self.m do
        if Qs[a] > maxQ then
          maxQ = Qs[a]
          bestAs = {a}
        elseif Qs[a] == maxQ then -- Ties can occur even with floats
          bestAs[#bestAs + 1] = a
        end
      end
      -- Perform random tie-breaking (if more than one argmax action)
      aIndex = bestAs[torch.random(1, #bestAs)]

      -- Compute saliency
      if self.saliency ~= 'none' then
        self:computeSaliency(state, aIndex)
      end
    end
  end

  -- If training
  if self.isTraining then
    -- Store experience tuple parts (including pre-emptive action)
    self.memory:store(reward, observation, terminal, aIndex)

    -- Collect validation transitions at the start
    if self.globals.step <= self.valSize then -- TODO: Collect enough *valid* transitions
      self.valMemory:store(reward, observation, terminal, aIndex)
    end

    -- Sample uniformly or with prioritised sampling
    if self.globals.step % self.memSampleFreq == 0 and self.globals.step >= self.learnStart then
      for n = 1, self.memNSamples do
        -- Optimise (learn) from experience tuples
        self:optimise(self.memory:sample(self.memPriority))
      end
    end

    -- Update target network every τ steps
    if self.globals.step % self.tau == 0 and self.globals.step >= self.learnStart then
      self.targetNet = self.policyNet:clone()
      self.targetNet:evaluate()
    end
  end

  -- Return action index with offset applied
  return aIndex - self.actionOffset
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
  local states, actions, rewards, transitions, terminals = self.memory:retrieve(indices) -- Terminal status is for transition (can't act in terminal state)

  -- Perform argmax action selection
  if self.doubleQ then
    -- Calculate Q-values from transition using policy network
    self.QPrimes = self.policyNet:forward(transitions) -- Find argmax actions using policy network
    -- Perform argmax action selection on transition using policy network: argmax_a[Q(s', a; θpolicy)]
    self.APrimeMax, self.APrimeMaxInds = torch.max(self.QPrimes, 2)
    -- Calculate Q-values from transition using target network
    self.QPrimes = self.targetNet:forward(transitions) -- Evaluate Q-values of argmax actions using target network
  else
    -- Calculate Q-values from transition using target network
    self.QPrimes = self.targetNet:forward(transitions) -- Find and evaluate Q-values of argmax actions using target network
    -- Perform argmax action selection on transition using target network: argmax_a[Q(s', a; θtarget)]
    self.APrimeMax, self.APrimeMaxInds = torch.max(self.QPrimes, 2)
  end

  -- Initially set target Y = Q(s', argmax_a[Q(s', a; θ)]; θtarget), where initial θ is either θtarget (DQN) or θpolicy (DDQN)
  for n = 1, self.batchSize do
    self.QPrimes[n]:mul(1 - terminals[n]) -- Zero Q(s' a) when s' is terminal
    self.Y[n] = self.QPrimes[n][self.APrimeMaxInds[n][1]]
  end
  -- Calculate target Y := r + γ.Q(s', argmax_a[Q(s', a; θ)]; θtarget)
  self.Y:mul(self.gamma):add(rewards)

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
    self.V = torch.max(self.Qs, 2) -- Current states cannot be terminal

    -- Calculate Advantage Learning update ∆ALQ(s, a) := δ − αPAL(V(s) − Q(s, a))
    self.tdErrAL = self.tdErr - self.V:add(-self.Q):mul(self.PALpha) -- TODO: Torch.CudaTensor:csub is missing

    -- Calculate Q(s', a) and V(s') using target network
    for n = 1, self.batchSize do
      self.QPrime[n] = self.QPrimes[n][actions[n]]
    end
    self.VPrime = torch.max(self.QPrimes, 2)

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
    self.QCurr[n][actions[n]] = ISWeights[n] * -self.tdErr[n] -- Negate target to use gradient descent (not ascent) optimisers
  end

  -- Backpropagate (network accumulates gradients internally)
  self.policyNet:backward(states, self.QCurr)
  -- Clip the L2 norm of the gradients
  if self.gradClip > 0 then
    self.policyNet:gradParamClip(self.gradClip)
  end

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
  if self.globals.step % self.progFreq == 0 then
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
  local startIndex, endIndex, batchSize, indices -- TODO: Use indices for *valid* validation transitions
  for n = 1, nBatches do
    startIndex = (n - 1)*self.batchSize + 1
    endIndex = n*self.batchSize
    batchSize = endIndex - startIndex + 1
    indices = torch.linspace(startIndex, endIndex, batchSize):long()

    -- Perform "learning" (without optimisation)
    self:learn(self.theta, indices, ISWeights:narrow(1, 1, batchSize))

    -- Calculate V(s') and TD-error δ
    if self.PALpha == 0 then
      self.VPrime = torch.max(self.QPrimes, 2)
    end
    totalV = totalV + torch.sum(self.VPrime)
    totalTdErr = totalTdErr + torch.abs(self.tdErr):sum()
  end

  -- Average and insert values
  self.avgV[#self.avgV + 1] = totalV / self.valSize
  self.avgTdErr[#self.avgTdErr + 1] = totalTdErr / self.valSize

  -- TODO Reduce memory consumption for gnuplot
  -- Plot losses
  gnuplot.pngfigure(paths.concat('experiments', self._id, 'losses.png'))
  gnuplot.plot('Loss', torch.linspace(math.floor(self.learnStart/self.progFreq), math.floor(self.globals.step/self.progFreq), #self.losses), torch.Tensor(self.losses), '-')
  gnuplot.xlabel('Step (x' .. self.progFreq .. ')')
  gnuplot.ylabel('Loss')
  gnuplot.plotflush()
  -- Plot V
  local epochIndices = torch.linspace(1, #self.avgV, #self.avgV)
  gnuplot.pngfigure(paths.concat('experiments', self._id, 'Vs.png'))
  gnuplot.plot('V', epochIndices, torch.Tensor(self.avgV), '-')
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('V')
  gnuplot.movelegend('left', 'top')
  gnuplot.plotflush()
  -- Plot TD-error δ
  gnuplot.pngfigure(paths.concat('experiments', self._id, 'TDErrors.png'))
  gnuplot.plot('TD-Error', epochIndices, torch.Tensor(self.avgTdErr), '-')
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('TD-Error')
  gnuplot.plotflush()
  -- Plot average score
  gnuplot.pngfigure(paths.concat('experiments', self._id, 'scores.png'))
  gnuplot.plot('Score', epochIndices, torch.Tensor(self.valScores), '-')
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('Average Score')
  gnuplot.movelegend('left', 'top')
  gnuplot.plotflush()

  return self.avgV[#self.avgV], self.avgTdErr[#self.avgTdErr]
end

-- Saves network convolutional filters as images
function Agent:visualiseFilters()
  local filters = self.model:getFilters()

  for i, v in ipairs(filters) do
    image.save(paths.concat('experiments', self._id, 'conv_layer_' .. i .. '.png'), v)
  end
end

-- Sets saliency style
function Agent:setSaliency(saliency)
  self.saliency = saliency
  self.model:setSaliency(saliency)
end

-- Computes a saliency map (assuming a forward pass of a single state)
function Agent:computeSaliency(state, index)
  -- Switch to possibly special backpropagation
  self.model:salientBackprop()

  -- Create artificial high target
  local maxTarget = self.Tensor(self.m):fill(0)
  maxTarget[index] = 2

  -- Backpropagate to inputs
  self.inputGrads = self.policyNet:backward(state, maxTarget)
  self.saliencyMap = image.scale(torch.abs(self.inputGrads:select(1, self.histLen):float()), self.origWidth, self.origHeight)

  -- Switch back to normal backpropagation
  self.model:normalBackprop()
end

-- Saves the network parameters θ
function Agent:saveWeights(path)
  torch.save(path, self.theta)
end

-- Loads network parameters θ
function Agent:loadWeights(path)
  self.theta = torch.load(path)
  self.targetNet = self.policyNet:clone()
  self.targetNet:evaluate()
end

return Agent
