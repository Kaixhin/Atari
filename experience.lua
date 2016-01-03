local _ = require 'moses'
local classic = require 'classic'
local BinaryHeap = require 'structures/BinaryHeap'
require 'classic.torch' -- Enables serialisation

local Experience = classic.class("Experience")

-- Converts a CDF from a PDF
local pdfToCdf = function(pdf)
  local c = 0
  pdf:apply(function(x)
    c = c + x
    return c
  end)
end

-- Creates experience replay memory
function Experience:_init(opt)
  -- Keep reference to opt
  self.opt = opt

  -- Create buffers
  self.buffers = {
    states = opt.Tensor(opt.batchSize, opt.histLen, opt.nChannels, opt.height, opt.width),
    actions = torch.ByteTensor(opt.batchSize),
    rewards = opt.Tensor(opt.batchSize),
    transitions = opt.Tensor(opt.batchSize, opt.histLen, opt.nChannels, opt.height, opt.width),
    terminals = torch.ByteTensor(opt.batchSize),
    priorities = opt.Tensor(opt.batchSize)
  }

  -- Allocate memory for experience
  local stateSize = torch.LongStorage({opt.memSize, opt.nChannels, opt.height, opt.width}) -- Calculate state storage size
  self.states = torch.FloatTensor(stateSize)
  self.actions = torch.ByteTensor(opt.memSize) -- Discrete action indices
  self.rewards = torch.FloatTensor(opt.memSize) -- Stored at time t
  -- Terminal conditions stored at time t+1, encoded by 0 = false, 1 = true
  self.terminals = torch.ByteTensor(opt.memSize):fill(1) -- Filling with 1 prevents going back in history initially
  -- Internal pointer
  self.index = 1
  self.isFull = false
  self.size = 0
  -- TD-error δ-based priorities
  self.priorityQueue = BinaryHeap(opt.memSize) -- Stored at time t
  self.smallConst = 1e-6 -- Account for half precision
  self.maxPriority = opt.tdClip -- Should prioritise sampling experience that has not been learnt from

  -- Work out tensor type to cast to for retrieveal
  self.castType = string.match(opt.Tensor():type(), 'torch.(.*)Tensor'):lower()

  -- Initialise first time step
  self.states[1]:zero() -- Blank out state
  self.actions[1] = 1 -- Action is no-op

  -- Calculates circular indices
  self.circIndex = function(x)
    local ind = x % opt.memSize
    return ind == 0 and opt.memSize or ind -- Correct 0-index
  end
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
  self.size = math.min(self.size + 1, self.opt.memSize)
  -- Circle back to beginning if memory limit reached
  if self.index > self.opt.memSize then
    self.isFull = true -- Full memory flag
    self.index = 1 -- Reset index
  end

  self.states[self.index] = state:float()
  self.terminals[self.index] = terminal and 1 or 0
  self.actions[self.index] = action
end

-- Retrieves experience tuples (s, a, r, s', t)
function Experience:retrieve(indices)
  -- Blank out history in one go
  self.buffers.states:zero()

  -- Iterate over indices
  for i = 1, self.opt.batchSize do
    local index = indices[i]
    -- Retrieve action
    self.buffers.actions[i] = self.actions[index]
    -- Retrieve rewards
    self.buffers.rewards[i] = self.rewards[index]
    -- Retrieve terminal status
    self.buffers.terminals[i] = self.terminals[index]

    -- Go back in history whilst episode exists
    local histIndex = self.opt.histLen
    repeat
      -- Copy state
      self.buffers.states[i][histIndex] = self.states[index][self.castType](self.states[index])
      -- Adjust indices
      index = self.circIndex(index - 1)
      histIndex = histIndex - 1
    until histIndex == 0 or self.terminals[index] == 1

    -- If not terminal, fill in transition history
    if self.buffers.terminals[i] == 0 then
      -- Copy most recent state
      for h = 2, self.opt.histLen do
        self.buffers.transitions[i][h] = self.buffers.states[i][h - 1]
      end
      -- Get transition frame
      self.buffers.transitions[i][self.opt.histLen] = self.states[self.circIndex(indices[i] + 1)][self.castType](self.states[index])
    end
  end

  return self.buffers.states, self.buffers.actions, self.buffers.rewards, self.buffers.transitions, self.buffers.terminals
end

-- Returns indices and importance-sampling weights based on (stochastic) proportional prioritised sampling
function Experience:sample(priorityType)
  local N = self.size
  local indices, w

  -- Priority 'none' = uniform sampling
  if priorityType == 'none' then
    indices = torch.randperm(N)[{{1, self.opt.batchSize}}]:long()
    w = torch.ones(self.opt.batchSize) -- Set weights to 1 as no correction needed
  elseif priorityType == 'rank' then
    -- Create table to store indices (by rank)
    local rankIndices = {} -- In reality the underlying array-based binary heap is used as an approximation of a ranked (sorted) array here
    -- Sample (rank) indices based on power-law distribution TODO: Cache partition indices for several values of N as α is static
    local samplingRange = torch.pow(1/self.opt.alpha, torch.linspace(1, math.log(N, 1/self.opt.alpha), self.opt.batchSize+1)):long() -- Use logarithmic binning
    -- Perform stratified sampling (transitions will have varying TD-error magnitudes |δ|)
    for i = 1, self.opt.batchSize do
      table.insert(rankIndices, torch.random(samplingRange[i], samplingRange[i+1]))
    end
    -- Retrieve actual transition indices
    indices = torch.LongTensor(self.priorityQueue:getValuesByVal(rankIndices))
    
    -- Importance-sampling weights w = (N * p(rank))^-β
    w = torch.Tensor(rankIndices):pow(-self.opt.alpha):mul(N):pow(-self.opt.beta[self.opt.step]) -- p(x) = Cx^-α but C is not analytical for α < 1 so this is unnormalised
    -- Find max importance-sampling weight for normalisation
    local wMax = torch.max(w) -- p(x) was unnormalised so again just use max of sample to normalise
    -- Normalise weights so updates only scale downwards (for stability)
    w:div(wMax) -- Max weight will be 1
  elseif priorityType == 'proportional' then
    --[[
    -- Calculate sampling probability distribution P
    local P = torch.pow(self.priorities[{{1, N}}], opt.alpha) -- Use prioritised experience replay exponent α
    local Z = torch.sum(P) -- Calculate normalisation constant
    P:div(Z) -- Normalise

    -- Calculate importance-sampling weights w
    w = torch.mul(P, N):pow(-opt.beta[opt.step]) -- Use importance-sampling exponent β
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

  if self.opt.gpu > 0 then
    w = w:cuda()
  end

  return indices, w
end

-- Update experience priorities using TD-errors δ
function Experience:updatePriorities(indices, delta)
  local priorities = delta:clone():float()
  if self.opt.memPriority == 'proportional' then
    priorities:abs()
  end

  for p = 1, indices:size(1) do
    self.priorityQueue:updateByVal(indices[p], priorities[p] + self.smallConst, indices[p]) -- Allows transitions to be sampled even if error is 0
  end
end

return Experience
