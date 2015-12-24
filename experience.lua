local _ = require 'moses'
require 'cutorch'

local experience = {}

-- Creates experience replay memory
experience.createMemory = function(length, stateSize)
  local memory = {}
  local stateSizes = torch.LongStorage(_.append({length}, stateSize)) -- Calculate state/transition storage size
  -- Allocate memory for experience
  local states = torch.Tensor(stateSizes)
  local actions = torch.Tensor(length)
  local rewards = torch.Tensor(length)
  local transitions = torch.Tensor(stateSizes)
  local terminals = torch.ByteTensor(length) -- Terminal conditions stored as 0 = false, 1 = true
  -- Internal pointer
  local nextIndex = 1
  local isFull = false

  -- Returns number of saved tuples
  memory.size = function()
    if isFull then
      return length
    else
      return nextIndex - 1
    end
  end

  -- Store new experience tuple
  memory.store = function(state, action, reward, transition, terminal)
    states[{{nextIndex}, {}}] = state:float()
    actions[nextIndex] = action
    rewards[nextIndex] = reward
    transitions[{{nextIndex}, {}}] = transition:float()
    terminals[nextIndex] = terminal and 1 or 0

    -- Increment index
    nextIndex = nextIndex + 1
    -- Circle back to beginning if memory limit reached
    if nextIndex > length then
      isFull = true -- Full memory flag
      nextIndex = 1 -- Reset nextIndex
    end
  end

  -- Retrieve experience tuples
  memory.retrieve = function(indices)
    return states:index(1, indices):cuda(), actions:index(1, indices), rewards:index(1, indices):cuda(), transitions:index(1, indices):cuda(), terminals:index(1, indices)
  end

  return memory
end

return experience
