local _ = require 'moses'
require 'cutorch'

local experience = {}

-- Creates experience replay memory
experience.createMemory = function(length, stateSize)
  local mem = {}
  local stateSizes = torch.LongStorage(_.append({length}, stateSize)) -- Calculate state/transition storage size
  -- Allocate memory for experience
  local states = torch.Tensor(stateSizes)
  local actions = torch.Tensor(length)
  local rewards = torch.Tensor(length)
  local transitions = torch.Tensor(stateSizes)
  -- Internal pointer
  local nextIndex = 1
  local isFull = false

  -- Returns number of saved tuples
  mem.size = function()
    if isFull then
      return length
    else
      return nextIndex - 1
    end
  end

  -- Store new experience tuple
  mem.store = function(s, a, r, t)
    states[{{nextIndex}, {}}] = s:float()
    actions[nextIndex] = a
    rewards[nextIndex] = r
    transitions[{{nextIndex}, {}}] = t:float()

    -- Increment index
    nextIndex = nextIndex + 1
    -- Circle back to beginning if memory limit reached
    if nextIndex > length then
      isFull = true -- Full memory flag
      nextIndex = 1 -- Reset nextIndex
    end
  end

  -- Retrieve experience tuples
  mem.retrieve = function(indices)
    return states:index(1, indices):cuda(), actions:index(1, indices), rewards:index(1, indices):cuda(), transitions:index(1, indices):cuda()
  end

  return mem
end

return experience
