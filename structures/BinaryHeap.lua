local classic = require 'classic'
require 'classic.torch' -- Enables serialisation

-- Implements a Priority Queue using a (Maximum) Binary Heap
local BinaryHeap = classic.class('BinaryHeap')

-- Creates a new Binary Heap with a length or existing tensor
function BinaryHeap:_init(init)
  if type(init) == 'number' then
    -- init is treated as the length of the heap
    self.array = torch.Tensor(init)
    self.size = 0
  else
    -- Otherwise assume tensor to build heap from
    self.array = init
    self.size = init:size(1)
    -- Rebalance
    for i = math.ceil(self.size/2) - 1, 1, -1 do
      self:rebalance(i)
    end
  end
end

--[[
-- Indices of connected nodes:
-- Parent(i) = floor(i/2)
-- Left_Child(i) = 2i
-- Right_Child(i) = 2i+1
--]]

-- Inserts a new value in place
function BinaryHeap:insert(val)
  -- Refuse to add values if no space left
  if self.size == self.array:size(1) then
    print('Error: no space left in heap to add value ' .. val)
    return
  end
  self.size = self.size + 1

  local i = self.size
  -- Bubble the value up from the bottom
  while i > 1 and self.array[math.floor(i/2)] < val do
    self.array[i], i = self.array[math.floor(i/2)], math.floor(i/2)
  end
  self.array[i] = val
end

-- Returns the maximum value
function BinaryHeap:peek()
  return self.size ~= 0 and self.array[1] or nil
end

-- Removes and returns the maximum value
function BinaryHeap:pop()
  -- Return nil if no values
  if self.size == 0 then
    print('Error: no values in heap')
    return nil
  end

  local max = self.array[1]

  -- Move the last value (not necessarily the smallest) to the root
  self.array[1] = self.array[self.size]
  self.size = self.size - 1
  -- Rebalance the tree
  self:rebalance(1)

  return max
end

-- Rebalances the heap
function BinaryHeap:rebalance(i)
  -- Calculate left and right child indices
  local l, r = 2*i, 2*i + 1

  -- Find the index of the greatest of these elements
  local greatest
  if l <= self.size and self.array[l] > self.array[i] then
    greatest = l
  else
    greatest = i
  end
  if r <= self.size and self.array[r] > self.array[greatest] then
    greatest = r
  end

  -- Continue rebalancing if necessary
  if greatest ~= i then
    self.array[i], self.array[greatest] = self.array[greatest], self.array[i]
    self:rebalance(greatest)
  end
end

-- Basic visualisation of heap
function BinaryHeap:__tostring()
  local str = ''
  local level = -1
  local maxLevel = math.floor(math.log(self.size, 2))
  
  -- Print each level
  for i = 1, self.size do
    -- Add a new line and spacing for each new level
    local l = math.floor(math.log(i, 2))
    if level ~= l then
      str = str .. '\n'
      level = l
    end
    -- Print value and spacing
    str = str .. string.format('%.2f ', self.array[i])
  end

  return str
end

-- Allow peeking at the internal array...
function BinaryHeap:__index(key)
  -- Returns values even if array "empty", but does allow {} index
  return self.array[key]
end

return BinaryHeap
