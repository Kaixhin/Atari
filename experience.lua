local _ = require 'moses'
local BinaryHeap = require 'structures/BinaryHeap'

local experience = {}

-- Converts a CDF from a PDF
local pdfToCdf = function(pdf)
  local c = 0
  pdf:apply(function(x)
    c = c + x
    return c
  end)
end

-- Creates experience replay memory
experience.create = function(opt)
  local memory = {}

  -- Create buffers
  experience.buffers = {
    states = opt.Tensor(opt.batchSize, opt.histLen, opt.nChannels, opt.height, opt.width),
    actions = torch.ByteTensor(opt.batchSize),
    rewards = opt.Tensor(opt.batchSize),
    transitions = opt.Tensor(opt.batchSize, opt.histLen, opt.nChannels, opt.height, opt.width),
    terminals = torch.ByteTensor(opt.batchSize)
  }

  -- Allocate memory for experience
  local stateSize = torch.LongStorage({opt.memSize, opt.nChannels, opt.height, opt.width}) -- Calculate state storage size
  memory.states = torch.FloatTensor(stateSize)
  memory.actions = torch.ByteTensor(opt.memSize) -- Discrete action indices
  memory.rewards = torch.FloatTensor(opt.memSize) -- Stored at time t
  -- Terminal conditions stored at time t+1, encoded by 0 = false, 1 = true
  memory.terminals = torch.ByteTensor(opt.memSize):fill(1) -- Filling with 1 prevents going back in history initially
  -- Internal pointer
  memory.index = 1
  memory.isFull = false
  -- TD-error δ-based priorities
  memory.priorities = torch.FloatTensor(opt.memSize):fill(0) -- Stored at time t
  local smallConst = 1e-6 -- Account for half precision
  memory.maxPriority = opt.tdClip -- Should prioritise sampling experience that has not been learnt from

  -- Work out tensor type to cast to for retrieveal
  local castType = string.match(opt.Tensor():type(), 'torch.(.*)Tensor'):lower()

  -- Initialise first time step
  memory.states[1]:zero() -- Blank out state
  memory.actions[1] = 1 -- Action is no-op

  -- Calculates circular indices
  local circIndex = function(x)
    local ind = x % opt.memSize
    return ind == 0 and opt.memSize or ind -- Correct 0-index
  end

  -- Returns number of saved tuples
  function memory:size()
    return self.isFull and opt.memSize or self.index - 1
  end

  -- Stores experience tuple parts (including pre-emptive action)
  function memory:store(reward, state, terminal, action)
    self.rewards[self.index] = reward
    -- Store with maximal priority
    self.priorities[self.index] = self.maxPriority + smallConst
    self.maxPriority = self.maxPriority + smallConst

    -- Increment index
    self.index = self.index + 1
    -- Circle back to beginning if memory limit reached
    if self.index > opt.memSize then
      self.isFull = true -- Full memory flag
      self.index = 1 -- Reset index
    end

    self.states[self.index] = state:float()
    self.terminals[self.index] = terminal and 1 or 0
    self.actions[self.index] = action
  end

  -- Retrieves experience tuples (s, a, r, s', t)
  function memory:retrieve(indices)
    -- Blank out history in one go
    experience.buffers.states:zero()

    -- Iterate over indices
    for i = 1, opt.batchSize do
      local index = indices[i]
      -- Retrieve action
      experience.buffers.actions[i] = self.actions[index]
      -- Retrieve rewards
      experience.buffers.rewards[i] = self.rewards[index]
      -- Retrieve terminal status
      experience.buffers.terminals[i] = self.terminals[index]

      -- Go back in history whilst episode exists
      local histIndex = opt.histLen
      repeat
        -- Copy state
        experience.buffers.states[i][histIndex] = self.states[index][castType](self.states[index])
        -- Adjust indices
        index = circIndex(index - 1)
        histIndex = histIndex - 1
      until histIndex == 0 or self.terminals[index] == 1

      -- If not terminal, fill in transition history
      if experience.buffers.terminals[i] == 0 then
        -- Copy most recent state
        for h = 2, opt.histLen do
          experience.buffers.transitions[i][h] = experience.buffers.states[i][h - 1]
        end
        -- Get transition frame
        experience.buffers.transitions[i][opt.histLen] = self.states[circIndex(indices[i] + 1)][castType](self.states[index])
      end
    end

    return experience.buffers.states, experience.buffers.actions, experience.buffers.rewards, experience.buffers.transitions, experience.buffers.terminals
  end

  -- Returns indices and importance-sampling weights based on (stochastic) proportional prioritised sampling
  function memory:sample(nSamples, priorityType)
    local N = self:size()
    local indices, w

    -- Priority 'none' = uniform sampling
    if priorityType == 'none' then
      indices = torch.randperm(N):long()
      indices = indices[{{1, nSamples}}]
      w = torch.ones(nSamples) -- Set weights to 1 as no correction needed
    else
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
    end

    if opt.gpu > 0 then
      w = w:cuda()
    end

    return indices, w
  end

  -- Update experience priorities using TD-errors δ
  function memory:updatePriorities(indices, delta)
    local priorities = delta:clone():float()
    if opt.memPriority == 'proportional' then
      priorities:abs()
    end

    for p = 1, indices:size(1) do
      self.priorities[indices[p]] = priorities[p] + smallConst -- Allows transitions to be sampled even if error is 0
    end
  end

  return memory
end

return experience
