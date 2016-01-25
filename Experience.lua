local _ = require 'moses'
local classic = require 'classic'
local BinaryHeap = require 'structures/BinaryHeap'
require 'classic.torch' -- Enables serialisation

local Experience = classic.class('Experience')

-- Converts a CDF from a PDF
local pdfToCdf = function(pdf)
  local c = 0
  pdf:apply(function(x)
    c = c + x
    return c
  end)
end

-- Creates experience replay memory
function Experience:_init(capacity, opt)
  self.capacity = capacity
  -- Extract relevant options
  self.batchSize = opt.batchSize
  self.histLen = opt.histLen
  self.gpu = opt.gpu
  self.memPriority = opt.memPriority
  self.alpha = opt.alpha
  self.betaZero = opt.betaZero
  -- Keep reference to opt for opt.step
  self.opt = opt -- TODO: Keep internal step counter

  -- Create transition tuples buffer
  self.transTuples = {
    states = opt.Tensor(opt.batchSize, opt.histLen, opt.nChannels, opt.height, opt.width),
    actions = torch.ByteTensor(opt.batchSize),
    rewards = opt.Tensor(opt.batchSize),
    transitions = opt.Tensor(opt.batchSize, opt.histLen, opt.nChannels, opt.height, opt.width),
    terminals = torch.ByteTensor(opt.batchSize),
    priorities = opt.Tensor(opt.batchSize)
  }

  -- Allocate memory for experience
  local stateSize = torch.LongStorage({capacity, opt.nChannels, opt.height, opt.width}) -- Calculate state storage size
  self.imgDiscLevels = 255 -- Number of discretisation levels for images (used for float <-> byte conversion)
  -- For the standard DQN problem, float vs. byte storage is 24GB vs. 6GB memory, so this prevents/minimises slow swap usage
  self.states = torch.ByteTensor(stateSize) -- ByteTensor to avoid massive memory usage
  self.actions = torch.ByteTensor(capacity) -- Discrete action indices
  self.rewards = torch.FloatTensor(capacity) -- Stored at time t (not t + 1)
  -- Terminal conditions stored at time t+1, encoded by 0 = false, 1 = true
  self.terminals = torch.ByteTensor(capacity):fill(1) -- Filling with 1 prevents going back in history initially
  -- Internal pointer
  self.index = 1
  self.isFull = false
  self.size = 0
  -- TD-error δ-based priorities
  self.priorityQueue = BinaryHeap(capacity) -- Stored at time t
  self.smallConst = 1e-6 -- Account for half precision
  self.maxPriority = opt.tdClip -- Should prioritise sampling experience that has not been learnt from

  -- Work out tensor type to cast to for retrieveal
  self.castType = string.match(opt.Tensor():type(), 'torch.(.*)Tensor'):lower()

  -- Initialise first time step
  self.states[1]:zero() -- Blank out state
  self.actions[1] = 1 -- Action is no-op

  -- Calculate β growth factor
  self.betaGrad = (1 - opt.betaZero)/opt.steps
end

-- Calculates circular indices
function Experience:circIndex(x)
  local ind = x % self.capacity
  return ind == 0 and self.capacity or ind -- Correct 0-index
end

-- Stores experience tuple parts (including pre-emptive action)
function Experience:store(reward, state, terminal, action)
  self.rewards[self.index] = reward
  -- Store with maximal priority
  self.maxPriority = self.maxPriority + self.smallConst
  if self.isFull then
    self.priorityQueue:updateByVal(self.index, self.maxPriority, self.index)
  else
    self.priorityQueue:insert(self.maxPriority, self.index)
  end

  -- Increment index and size
  self.index = self.index + 1
  self.size = math.min(self.size + 1, self.capacity)
  -- Circle back to beginning if memory limit reached
  if self.index > self.capacity then
    self.isFull = true -- Full memory flag
    self.index = 1 -- Reset index
  end

  self.states[self.index] = state:float():mul(self.imgDiscLevels) -- float -> byte
  self.terminals[self.index] = terminal and 1 or 0
  self.actions[self.index] = action
end

-- Retrieves experience tuples (s, a, r, s', t)
function Experience:retrieve(indices)
  local batchSize = indices:size(1)
  -- Blank out history in one go
  self.transTuples.states:zero()

  -- Iterate over indices
  for i = 1, batchSize do
    local memIndex = indices[i]
    -- Retrieve action
    self.transTuples.actions[i] = self.actions[memIndex]
    -- Retrieve rewards
    self.transTuples.rewards[i] = self.rewards[memIndex]
    -- Retrieve terminal status
    self.transTuples.terminals[i] = self.terminals[memIndex]

    -- Go back in history whilst episode exists
    local histIndex = self.histLen
    repeat
      -- Copy state
      self.transTuples.states[i][histIndex] = self.states[memIndex][self.castType](self.states[memIndex]):div(self.imgDiscLevels) -- byte -> float
      -- Adjust indices
      memIndex = self:circIndex(memIndex - 1)
      histIndex = histIndex - 1
    until histIndex == 0 or self.terminals[memIndex] == 1

    -- If not terminal, fill in transition history
    if self.transTuples.terminals[i] == 0 then
      -- Copy most recent state
      for h = 2, self.histLen do
        self.transTuples.transitions[i][h - 1] = self.transTuples.states[i][h]
      end
      -- Get transition frame
      self.transTuples.transitions[i][self.histLen] = self.states[self:circIndex(indices[i] + 1)][self.castType](self.states[memIndex]):div(self.imgDiscLevels) -- byte -> float
    end
  end

  return self.transTuples.states, self.transTuples.actions, self.transTuples.rewards, self.transTuples.transitions, self.transTuples.terminals
end

-- Returns indices and importance-sampling weights based on (stochastic) proportional prioritised sampling
function Experience:sample(priorityType)
  local N = self.size
  local indices, w

  -- Priority 'none' = uniform sampling
  if priorityType == 'none' then
    indices = torch.randperm(N)[{{1, self.batchSize}}]:long()
    w = torch.ones(self.batchSize) -- Set weights to 1 as no correction needed
  elseif priorityType == 'rank' then
    -- Create table to store indices (by rank)
    local rankIndices = {} -- In reality the underlying array-based binary heap is used as an approximation of a ranked (sorted) array here
    -- Sample (rank) indices based on power-law distribution TODO: Cache partition indices for several values of N as α is static
    local samplingRange = torch.pow(1/self.alpha, torch.linspace(1, math.log(N, 1/self.alpha), self.batchSize+1)):long() -- Use logarithmic binning
    -- Perform stratified sampling (transitions will have varying TD-error magnitudes |δ|)
    for i = 1, self.batchSize do
      rankIndices[#rankIndices + 1] = torch.random(samplingRange[i], samplingRange[i+1])
    end
    -- Retrieve actual transition indices
    indices = torch.LongTensor(self.priorityQueue:getValuesByVal(rankIndices))
    
    -- Importance-sampling weights w = (N * p(rank))^-β
    local beta = math.min(self.betaZero + (self.opt.step - 1)*self.betaGrad, 1)
    w = torch.Tensor(rankIndices):pow(-self.alpha):mul(N):pow(-beta) -- p(x) = Cx^-α but C is not analytical for α < 1 so this is unnormalised
    -- Find max importance-sampling weight for normalisation
    local wMax = torch.max(w) -- p(x) was unnormalised so again just use max of sample to normalise
    -- Normalise weights so updates only scale downwards (for stability)
    w:div(wMax) -- Max weight will be 1
  elseif priorityType == 'proportional' then
    --[[
    -- Calculate sampling probability distribution P
    local P = torch.pow(self.priorities[{{1, N}}], self.alpha) -- Use prioritised experience replay exponent α
    local Z = torch.sum(P) -- Calculate normalisation constant
    P:div(Z) -- Normalise

    -- Calculate importance-sampling weights w
    local beta = math.min(self.betaZero + (self.opt.step - 1)*self.betaGrad, 1)
    w = torch.mul(P, N):pow(-beta) -- Use importance-sampling exponent β
    w:div(torch.max(w)) -- Normalise weights so updates only scale downwards (for stability)

    -- Create a cumulative distribution for inverse transform sampling
    pdfToCdf(P) -- Convert distribution
    indices = torch.sort(torch.Tensor(nSamples):uniform()) -- Generate uniform numbers for sampling
    -- Perform linear search to sample
    local minIndex = 1
    for i = 1, nSamples do
      while indices[i] > P[minIndex] do
        minIndex = minIndex + 1
      end
      indices[i] = minIndex -- Get sampled index
    end
    indices = indices:long() -- Convert to LongTensor for indexing
    w = w:index(1, indices) -- Index weights
    --]]
  end

  if self.gpu > 0 then
    w = w:cuda()
  end

  return indices, w
end

-- Update experience priorities using TD-errors δ
function Experience:updatePriorities(indices, delta)
  local priorities = delta:clone():float()
  if self.memPriority == 'proportional' then
    priorities:abs()
  end

  for p = 1, indices:size(1) do
    self.priorityQueue:updateByVal(indices[p], priorities[p] + self.smallConst, indices[p]) -- Allows transitions to be sampled even if error is 0
  end
end

return Experience
