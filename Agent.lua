local _ = require 'moses'
local class = require 'classic'
local optim = require 'optim'
local gnuplot = require 'gnuplot'
local Model = require 'Model'
local Experience = require 'Experience'
local CircularQueue = require 'structures/CircularQueue'
local Singleton = require 'structures/Singleton'
local AbstractAgent = require 'async/AbstractAgent'
require 'classic.torch' -- Enables serialisation
require 'modules/rmspropm' -- Add RMSProp with momentum

-- Detect QT for image display
local qt = pcall(require, 'qt')

local Agent = classic.class('Agent', AbstractAgent)

-- Creates a DQN agent
function Agent:_init(opt)
  -- Experiment ID
  self._id = opt._id
  self.experiments = opt.experiments
  -- Actions
  self.m = opt.actionSpec[3][2] - opt.actionSpec[3][1] + 1 -- Number of discrete actions
  self.actionOffset = 1 - opt.actionSpec[3][1] -- Calculate offset if first action is not indexed as 1

  -- Initialise model helper
  self.model = Model(opt)
  -- Create policy and target networks
  self.policyNet = self.model:create()
  self.targetNet = self.policyNet:clone() -- Create deep copy for target network
  self.targetNet:evaluate() -- Target network always in evaluation mode
  self.tau = opt.tau
  self.doubleQ = opt.doubleQ
  -- Network parameters θ and gradients dθ
  self.theta, self.dTheta = self.policyNet:getParameters()

  -- Boostrapping
  self.bootstraps = opt.bootstraps
  self.head = 1 -- Identity of current episode bootstrap head
  self.heads = math.max(opt.bootstraps, 1) -- Number of heads

  -- Recurrency
  self.recurrent = opt.recurrent
  self.histLen = opt.histLen

  -- Reinforcement learning parameters
  self.gamma = opt.gamma
  self.rewardClip = opt.rewardClip
  self.tdClip = opt.tdClip
  self.epsilonStart = opt.epsilonStart
  self.epsilonEnd = opt.epsilonEnd
  self.epsilonGrad = (opt.epsilonEnd - opt.epsilonStart)/opt.epsilonSteps -- Greediness ε decay factor
  self.PALpha = opt.PALpha

  -- State buffer
  self.stateBuffer = CircularQueue(opt.recurrent and 1 or opt.histLen, opt.Tensor, opt.stateSpec[2])
  -- Experience replay memory
  self.memory = Experience(opt.memSize, opt)
  self.memSampleFreq = opt.memSampleFreq
  self.memNSamples = opt.memNSamples
  self.memSize = opt.memSize
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

  -- Q-learning variables (per head)
  self.QPrimes = opt.Tensor(opt.batchSize, self.heads, self.m)
  self.tdErr = opt.Tensor(opt.batchSize, self.heads)
  self.VPrime = opt.Tensor(opt.batchSize, self.heads, 1)

  -- Validation variables
  self.valSize = opt.valSize
  self.valMemory = Experience(opt.valSize + 3, opt, true) -- Validation experience replay memory (with empty starting state...states...final transition...blank state)
  self.losses = {}
  self.avgV = {} -- Running average of V(s')
  self.avgTdErr = {} -- Running average of TD-error δ
  self.valScores = {} -- Validation scores (passed from main script)
  self.normScores = {} -- Normalised validation scores (passed from main script)

  -- Tensor creation
  self.Tensor = opt.Tensor

  -- Saliency display
  self:setSaliency(opt.saliency) -- Set saliency option on agent and model
  if #opt.stateSpec[2] == 3 then -- Make saliency map only for visual states
    self.saliencyMap = opt.Tensor(1, opt.stateSpec[2][2], opt.stateSpec[2][3]):zero()
    self.inputGrads = opt.Tensor(opt.histLen*opt.stateSpec[2][1], opt.stateSpec[2][2], opt.stateSpec[2][3]):zero() -- Gradients with respect to the input (for saliency maps)
  end

  -- Get singleton instance for step
  self.globals = Singleton.getInstance()
end

-- Sets training mode
function Agent:training()
  self.isTraining = true
  self.policyNet:training()
  -- Clear state buffer
  self.stateBuffer:clear()
  -- Reset bootstrap head
  if self.bootstraps > 0 then
    self.head = torch.random(self.bootstraps)
  end
  -- Forget last sequence
  if self.recurrent then
    self.policyNet:forget()
    self.targetNet:forget()
  end
end

-- Sets evaluation mode
function Agent:evaluate()
  self.isTraining = false
  self.policyNet:evaluate()
  -- Clear state buffer
  self.stateBuffer:clear()
  -- Set previously stored state as invalid (as no transition stored)
  self.memory:setInvalid()
  -- Reset bootstrap head
  if self.bootstraps > 0 then
    self.head = torch.random(self.bootstraps)
  end
  -- Forget last sequence
  if self.recurrent then
    self.policyNet:forget()
  end
end
  
-- Observes the results of the previous transition and chooses the next action to perform
function Agent:observe(reward, rawObservation, terminal)
  -- Clip reward for stability
  if self.rewardClip > 0 then
    reward = math.max(reward, -self.rewardClip)
    reward = math.min(reward, self.rewardClip)
  end

  -- Process observation of current state
  local observation = self.model:preprocess(rawObservation) -- Must avoid side-effects on observation from env

  -- Store in buffer depending on terminal status
  if terminal then
    self.stateBuffer:pushReset(observation) -- Will clear buffer on next push
  else
    self.stateBuffer:push(observation)
  end
  -- Retrieve current and historical states from state buffer
  local state = self.stateBuffer:readAll()

  -- Set ε based on training vs. evaluation mode
  local epsilon = 0.001 -- Taken from tuned DDQN evaluation
  if self.isTraining then
    if self.globals.step < self.learnStart then
      -- Keep ε constant before learning starts
      epsilon = self.epsilonStart
    else
      -- Use annealing ε
      epsilon = math.max(self.epsilonStart + (self.globals.step - self.learnStart - 1)*self.epsilonGrad, self.epsilonEnd)
    end
  end

  local aIndex = 1 -- In a terminal state, choose no-op/first action by default
  if not terminal then
    if not self.isTraining and self.bootstraps > 0 then
      -- Retrieve estimates from all heads
      local QHeads = self.policyNet:forward(state)

      -- Use ensemble policy with bootstrap heads (in evaluation mode)
      local QHeadsMax, QHeadsMaxInds = QHeads:max(2) -- Find max action per head
      aIndex = torch.mode(QHeadsMaxInds:float(), 1)[1][1] -- TODO: Torch.CudaTensor:mode is missing

      -- Plot uncertainty in ensemble policy
      if qt then
        gnuplot.hist(QHeadsMaxInds, self.m, 0.5, self.m + 0.5)
      end

      -- Compute saliency map
      if self.saliency then
        self:computeSaliency(state, aIndex, true)
      end
    elseif torch.uniform() < epsilon then 
      -- Choose action by ε-greedy exploration (even with bootstraps)
      aIndex = torch.random(1, self.m)

      -- Forward state anyway if recurrent
      if self.recurrent then
        self.policyNet:forward(state)
      end

      -- Reset saliency if action not chosen by network
      if self.saliency then
        self.saliencyMap:zero()
      end
    else
      -- Retrieve estimates from all heads
      local QHeads = self.policyNet:forward(state)

      -- Sample from current episode head (indexes on first dimension with no batch)
      local Qs = QHeads:select(1, self.head)
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
      if self.saliency then
        self:computeSaliency(state, aIndex, false)
      end
    end
  end

  -- If training
  if self.isTraining then
    -- Store experience tuple parts (including pre-emptive action)
    self.memory:store(reward, observation, terminal, aIndex) -- TODO: Sample independent Bernoulli(p) bootstrap masks for all heads; p = 1 means no masks needed

    -- Collect validation transitions at the start
    if self.globals.step <= self.valSize + 1 then
      self.valMemory:store(reward, observation, terminal, aIndex)
    end

    -- Sample uniformly or with prioritised sampling
    if self.globals.step % self.memSampleFreq == 0 and self.globals.step >= self.learnStart then
      for n = 1, self.memNSamples do
        -- Optimise (learn) from experience tuples
        self:optimise(self.memory:sample())
      end
    end

    -- Update target network every τ steps
    if self.globals.step % self.tau == 0 and self.globals.step >= self.learnStart then
      self.targetNet = self.policyNet:clone()
      self.targetNet:evaluate()
    end

    -- Rebalance priority queue for prioritised experience replay
    if self.globals.step % self.memSize == 0 and self.memPriority then
      self.memory:rebalance()
    end
  end

  if terminal then
    if self.bootstraps > 0 then
      -- Change bootstrap head for next episode
      self.head = torch.random(self.bootstraps)
    elseif self.recurrent then
      -- Forget last sequence
      self.policyNet:forget()
    end
  end

  -- Return action index with offset applied
  return aIndex - self.actionOffset
end

-- Learns from experience
function Agent:learn(x, indices, ISWeights, isValidation)
  -- Copy x to parameters θ if necessary
  if x ~= self.theta then
    self.theta:copy(x)
  end
  -- Reset gradients dθ
  self.dTheta:zero()

  -- Retrieve experience tuples
  local memory = isValidation and self.valMemory or self.memory
  local states, actions, rewards, transitions, terminals = memory:retrieve(indices) -- Terminal status is for transition (can't act in terminal state)
  local N = actions:size(1)

  if self.recurrent then
    -- Forget last sequence
    self.policyNet:forget()
    self.targetNet:forget()
  end

  -- Perform argmax action selection
  local APrimeMax, APrimeMaxInds
  if self.doubleQ then
    -- Calculate Q-values from transition using policy network
    self.QPrimes = self.policyNet:forward(transitions) -- Find argmax actions using policy network
    -- Perform argmax action selection on transition using policy network: argmax_a[Q(s', a; θpolicy)]
    APrimeMax, APrimeMaxInds = torch.max(self.QPrimes, 3)
    -- Calculate Q-values from transition using target network
    self.QPrimes = self.targetNet:forward(transitions) -- Evaluate Q-values of argmax actions using target network
  else
    -- Calculate Q-values from transition using target network
    self.QPrimes = self.targetNet:forward(transitions) -- Find and evaluate Q-values of argmax actions using target network
    -- Perform argmax action selection on transition using target network: argmax_a[Q(s', a; θtarget)]
    APrimeMax, APrimeMaxInds = torch.max(self.QPrimes, 3)
  end

  -- Initially set target Y = Q(s', argmax_a[Q(s', a; θ)]; θtarget), where initial θ is either θtarget (DQN) or θpolicy (DDQN)
  local Y = self.Tensor(N, self.heads)
  for n = 1, N do
    self.QPrimes[n]:mul(1 - terminals[n]) -- Zero Q(s' a) when s' is terminal
    Y[n] = self.QPrimes[n]:gather(2, APrimeMaxInds[n])
  end
  -- Calculate target Y := r + γ.Q(s', argmax_a[Q(s', a; θ)]; θtarget)
  Y:mul(self.gamma):add(rewards:repeatTensor(1, self.heads))

  -- Get all predicted Q-values from the current state
  if self.recurrent and self.doubleQ then
    self.policyNet:forget()
  end
  local QCurr = self.policyNet:forward(states) -- Correct internal state of policy network before backprop
  local QTaken = self.Tensor(N, self.heads)
  -- Get prediction of current Q-values with given actions
  for n = 1, N do
    QTaken[n] = QCurr[n][{{}, {actions[n]}}]
  end

  -- Calculate TD-errors δ := ∆Q(s, a) = Y − Q(s, a)
  self.tdErr = Y - QTaken

  -- Calculate Advantage Learning update(s)
  if self.PALpha > 0 then
    -- Calculate Q(s, a) and V(s) using target network
    if self.recurrent then
      self.targetNet:forget()
    end
    local Qs = self.targetNet:forward(states)
    local Q = self.Tensor(N, self.heads)
    for n = 1, N do
      Q[n] = Qs[n][{{}, {actions[n]}}]
    end
    local V = torch.max(Qs, 3) -- Current states cannot be terminal

    -- Calculate Advantage Learning update ∆ALQ(s, a) := δ − αPAL(V(s) − Q(s, a))
    local tdErrAL = self.tdErr - V:csub(Q):mul(self.PALpha)

    -- Calculate Q(s', a) and V(s') using target network
    local QPrime = self.Tensor(N, self.heads)
    for n = 1, N do
      QPrime[n] = self.QPrimes[n][{{}, {actions[n]}}]
    end
    self.VPrime = torch.max(self.QPrimes, 3)

    -- Calculate Persistent Advantage Learning update ∆PALQ(s, a) := max[∆ALQ(s, a), δ − αPAL(V(s') − Q(s', a))]
    self.tdErr = torch.max(torch.cat(tdErrAL, self.tdErr:csub((self.VPrime:csub(QPrime):mul(self.PALpha))), 3), 3):view(N, 1)
  end

  -- Calculate loss
  local loss
  if self.tdClip > 0 then
    -- Squared loss is used within clipping range, absolute loss is used outside (approximates Huber loss)
    local sqLoss = torch.cmin(torch.abs(self.tdErr), self.tdClip)
    local absLoss = torch.abs(self.tdErr) - sqLoss
    loss = torch.mean(sqLoss:pow(2):mul(0.5):add(absLoss:mul(self.tdClip))) -- Average over heads

    -- Clip TD-errors δ
    self.tdErr:clamp(-self.tdClip, self.tdClip)
  else
    -- Squared loss
    loss = torch.mean(self.tdErr:clone():pow(2):mul(0.5)) -- Average over heads
  end

  -- Exit if being used for validation metrics
  if isValidation then
    return
  end

  -- Send TD-errors δ to be used as priorities
  self.memory:updatePriorities(indices, torch.mean(self.tdErr, 2)) -- Use average error over heads
  
  -- Zero QCurr outputs (no error)
  QCurr:zero()
  -- Set TD-errors δ with given actions
  for n = 1, N do
    -- Correct prioritisation bias with importance-sampling weights
    QCurr[n][{{}, {actions[n]}}] = torch.mul(-self.tdErr[n], ISWeights[n]) -- Negate target to use gradient descent (not ascent) optimisers
  end

  -- Backpropagate (network accumulates gradients internally)
  self.policyNet:backward(states, QCurr) -- TODO: Work out why DRQN crashes on different batch sizes
  -- Clip the L2 norm of the gradients
  if self.gradClip > 0 then
    self.policyNet:gradParamClip(self.gradClip)
  end

  if self.recurrent then
    -- Forget last sequence
    self.policyNet:forget()
    self.targetNet:forget()
    -- Previous hidden state of policy net not restored as model parameters changed
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

-- Pretty prints array
local pprintArr = function(memo, v)
  return memo .. ', ' .. v
end

-- Reports absolute network weights and gradients
function Agent:report()
  -- Collect layer with weights
  local weightLayers = self.policyNet:findModules('nn.SpatialConvolution')
  local fcLayers = self.policyNet:findModules('nn.Linear')
  weightLayers = _.append(weightLayers, fcLayers)
  
  -- Array of norms and maxima
  local wNorms = {}
  local wMaxima = {}
  local wGradNorms = {}
  local wGradMaxima = {}

  -- Collect statistics
  for l = 1, #weightLayers do
    local w = weightLayers[l].weight:clone():abs() -- Weights (absolute)
    wNorms[#wNorms + 1] = torch.mean(w) -- Weight norms:
    wMaxima[#wMaxima + 1] = torch.max(w) -- Weight max
    w = weightLayers[l].gradWeight:clone():abs() -- Weight gradients (absolute)
    wGradNorms[#wGradNorms + 1] = torch.mean(w) -- Weight grad norms:
    wGradMaxima[#wGradMaxima + 1] = torch.max(w) -- Weight grad max
  end

  -- Create report string table
  local reports = {
    'Weight norms: ' .. _.reduce(wNorms, pprintArr),
    'Weight max: ' .. _.reduce(wMaxima, pprintArr),
    'Weight gradient norms: ' .. _.reduce(wGradNorms, pprintArr),
    'Weight gradient max: ' .. _.reduce(wGradMaxima, pprintArr)
  }

  return reports
end

-- Reports stats for validation
function Agent:validate()
  -- Validation variables
  local totalV, totalTdErr = 0, 0

  -- Loop over validation transitions
  local nBatches = math.ceil(self.valSize / self.batchSize)
  local ISWeights = self.Tensor(self.batchSize):fill(1)
  local startIndex, endIndex, batchSize, indices
  for n = 1, nBatches do
    startIndex = (n - 1)*self.batchSize + 2
    endIndex = math.min(n*self.batchSize + 1, self.valSize + 1)
    batchSize = endIndex - startIndex + 1
    indices = torch.linspace(startIndex, endIndex, batchSize):long()

    -- Perform "learning" (without optimisation)
    self:learn(self.theta, indices, ISWeights:narrow(1, 1, batchSize), true)

    -- Calculate V(s') and TD-error δ
    if self.PALpha == 0 then
      self.VPrime = torch.max(self.QPrimes, 3)
    end
    -- Average over heads
    totalV = totalV + torch.mean(self.VPrime, 2):sum()
    totalTdErr = totalTdErr + torch.mean(self.tdErr, 2):abs():sum()
  end

  -- Average and insert values
  self.avgV[#self.avgV + 1] = totalV / self.valSize
  self.avgTdErr[#self.avgTdErr + 1] = totalTdErr / self.valSize

  -- TODO: Reduce memory consumption for gnuplot
  -- Plot and save losses
  if #self.losses > 0 then
    local losses = torch.Tensor(self.losses)
    gnuplot.pngfigure(paths.concat(self.experiments, self._id, 'losses.png'))
    gnuplot.plot('Loss', torch.linspace(math.floor(self.learnStart/self.progFreq), math.floor(self.globals.step/self.progFreq), #self.losses), losses, '-')
    gnuplot.xlabel('Step (x' .. self.progFreq .. ')')
    gnuplot.ylabel('Loss')
    gnuplot.plotflush()
    torch.save(paths.concat(self.experiments, self._id, 'losses.t7'), losses)
  end
  -- Plot and save V
  local epochIndices = torch.linspace(1, #self.avgV, #self.avgV)
  local Vs = torch.Tensor(self.avgV)
  gnuplot.pngfigure(paths.concat(self.experiments, self._id, 'Vs.png'))
  gnuplot.plot('V', epochIndices, Vs, '-')
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('V')
  gnuplot.movelegend('left', 'top')
  gnuplot.plotflush()
  torch.save(paths.concat(self.experiments, self._id, 'V.t7'), Vs)
  -- Plot and save TD-error δ
  local TDErrors = torch.Tensor(self.avgTdErr)
  gnuplot.pngfigure(paths.concat(self.experiments, self._id, 'TDErrors.png'))
  gnuplot.plot('TD-Error', epochIndices, TDErrors, '-')
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('TD-Error')
  gnuplot.plotflush()
  torch.save(paths.concat(self.experiments, self._id, 'TDErrors.t7'), TDErrors)
  -- Plot and save average score
  local scores = torch.Tensor(self.valScores)
  gnuplot.pngfigure(paths.concat(self.experiments, self._id, 'scores.png'))
  gnuplot.plot('Score', epochIndices, scores, '-')
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('Average Score')
  gnuplot.movelegend('left', 'top')
  gnuplot.plotflush()
  torch.save(paths.concat(self.experiments, self._id, 'scores.t7'), scores)
    -- Plot and save normalised score
  if #self.normScores > 0 then
    local normScores = torch.Tensor(self.normScores)
    gnuplot.pngfigure(paths.concat(self.experiments, self._id, 'normScores.png'))
    gnuplot.plot('Score', epochIndices, normScores, '-')
    gnuplot.xlabel('Epoch')
    gnuplot.ylabel('Normalised Score')
    gnuplot.movelegend('left', 'top')
    gnuplot.plotflush()
    torch.save(paths.concat(self.experiments, self._id, 'normScores.t7'), normScores)
  end

  return self.avgV[#self.avgV], self.avgTdErr[#self.avgTdErr]
end

-- Saves network convolutional filters as images
function Agent:visualiseFilters()
  local filters = self.model:getFilters()

  for i, v in ipairs(filters) do
    image.save(paths.concat(self.experiments, self._id, 'conv_layer_' .. i .. '.png'), v)
  end
end

-- Sets saliency style
function Agent:setSaliency(saliency)
  self.saliency = saliency
  self.model:setSaliency(saliency)
end

-- Computes a saliency map (assuming a forward pass of a single state)
function Agent:computeSaliency(state, index, ensemble)
  -- Switch to possibly special backpropagation
  self.model:salientBackprop()

  -- Create artificial high target
  local maxTarget = self.Tensor(self.heads, self.m):zero()
  if ensemble then
    -- Set target on all heads (when using ensemble policy)
    maxTarget[{{}, {index}}] = 1
  else
    -- Set target on current head
    maxTarget[self.head][index] = 1
  end

  -- Backpropagate to inputs
  self.inputGrads = self.policyNet:backward(state, maxTarget)
  -- Saliency map ref used by Display
  self.saliencyMap = torch.abs(self.inputGrads:select(1, self.recurrent and 1 or self.histLen):float())

  -- Switch back to normal backpropagation
  self.model:normalBackprop()
end

-- Saves the network parameters θ
function Agent:saveWeights(path)
  torch.save(path, self.theta:float()) -- Do not save as CudaTensor to increase compatibility
end

-- Loads network parameters θ
function Agent:loadWeights(path)
  local weights = torch.load(path)
  self.theta:copy(weights)
  self.targetNet = self.policyNet:clone()
  self.targetNet:evaluate()
end

return Agent
