local _ = require 'moses'
require 'cutorch'

local experience = {}

-- Creates experience replay memory
experience.create = function(stateSize, opt)
  local memory = {}
  local stateSizes = torch.LongStorage(_.append({opt.memSize}, stateSize)) -- Calculate state/transition storage size
  -- Allocate memory for experience
  memory.states = torch.Tensor(stateSizes)
  memory.actions = torch.Tensor(opt.memSize)
  memory.rewards = torch.Tensor(opt.memSize)
  memory.transitions = torch.Tensor(stateSizes)
  memory.terminals = torch.ByteTensor(opt.memSize) -- Terminal conditions stored as 0 = false, 1 = true
  -- Internal pointer
  memory.nextIndex = 1
  memory.isFull = false
  -- TD-error δ-based priorities
  memory.priorities = torch.Tensor(opt.memSize)
  local smallConst = 1e-9
  memory.maxPriority = opt.tdClamp -- Should prioritise sampling experience that has not been learnt from

  -- Returns number of saved tuples
  memory.size = function(self)
    if self.isFull then
      return opt.memSize
    else
      return self.nextIndex - 1
    end
  end

  -- Store new experience tuple
  memory.store = function(self, state, action, reward, transition, terminal)
    self.states[{{self.nextIndex}, {}}] = state:float()
    self.actions[self.nextIndex] = action
    self.rewards[self.nextIndex] = reward
    self.transitions[{{self.nextIndex}, {}}] = transition:float()
    self.terminals[self.nextIndex] = terminal and 1 or 0
    -- Store with maximal priority
    self.priorities[self.nextIndex] = self.maxPriority + smallConst
    self.maxPriority = self.maxPriority + smallConst

    -- Increment index
    self.nextIndex = self.nextIndex + 1
    -- Circle back to beginning if memory limit reached
    if self.nextIndex > opt.memSize then
      self.isFull = true -- Full memory flag
      self.nextIndex = 1 -- Reset nextIndex
    end
  end

  -- Retrieve experience tuples
  memory.retrieve = function(self, indices)
    return self.states:index(1, indices):cuda(), self.actions:index(1, indices), self.rewards:index(1, indices):cuda(), self.transitions:index(1, indices):cuda(), self.terminals:index(1, indices)
  end

  -- Update experience priorities
  memory.updatePriorities = function(self, indices, priorities)
    for p = 1, indices:size(1) do
      self.priorities[indices[p]] = priorities[p] + smallConst -- Allows transitions to be sampled even if error is 0
    end
  end

  -- Retrieve experience priorities
  memory.retrievePriorities = function(self, indices)
    return self.priorities:index(1, indices)
  end

  -- Converts a CDF from a PDF
  local pdfToCdf = function(pdf)
    local c = 0
    pdf:apply(function(x)
      c = c + x
      return c
    end)
  end

  -- Returns indices and importance-sampling weights based on (stochastic) proportional prioritised sampling
  memory.prioritySample = function(self, sampleSize)
    local N = self:size()

    -- Calculate sampling probability distribution P
    local expPriorities = torch.pow(self.priorities[{{1, N}}], opt.alpha) -- Use prioritised experience replay exponent α
    local Z = torch.sum(expPriorities) -- Normalisation constant
    local P = expPriorities:div(Z)

    -- Calculate importance-sampling weights w
    local w = torch.pow(torch.mul(P, N), -opt.beta[opt.step]) -- Use importance-sampling exponent β
    w:div(torch.max(w)) -- Normalise weights so updates only scale downwards (for stability)

    -- Create a cumulative distribution for inverse transform sampling
    pdfToCdf(P) -- Convert distribution
    local indices = torch.sort(torch.Tensor(sampleSize):uniform()) -- Generate uniform numbers for sampling
    -- Perform linear search to sample
    local minIndex = 1
    for i = 1, sampleSize do
      while indices[i] > P[minIndex] do
        minIndex = minIndex + 1
      end
      indices[i] = minIndex -- Get sampled index
    end
    indices = indices:long() -- Convert to LongTensor for indexing

    return indices, w:index(1, indices)
  end

  return memory
end

return experience
