local classic = require 'classic'
require 'classic.torch' -- Enables serialisation

-- Implements a Priority Queue using a non-standard (Maximum) Binary Heap
local BinaryHeap = classic.class('BinaryHeap')

-- Creates a new Binary Heap with a length or existing tensor
function BinaryHeap:_init(init)
  -- Use values as indices in a hash table
  self.hash = {}

  if type(init) == 'number' then
    -- init is treated as the length of the heap
    self.array = torch.Tensor(init, 2) -- Priorities are 1st, values (which are used as hash table keys) are 2nd
    self.size = 0
  else
    -- Otherwise assume tensor to build heap from
    self.array = init
    self.size = init:size(1)
    -- Rebalance
    for i = math.ceil(self.size/2) - 1, 1, -1 do
      self:downHeap(i)
    end
  end
end

--[[
-- Indices of connected nodes:
-- Parent(i) = floor(i/2)
-- Left_Child(i) = 2i
-- Right_Child(i) = 2i+1
--]]

-- Inserts a new value
function BinaryHeap:insert(priority, val)
  -- Refuse to add values if no space left
  if self.size == self.array:size(1) then
    print('Error: no space left in heap to add value ' .. val .. ' with priority ' .. priority)
    return
  end

  -- Add value to end
  self.size = self.size + 1
  self.array[self.size][1] = priority
  self.array[self.size][2] = val
  -- Update hash table
  self.hash[val] = self.size

  -- Rebalance
  self:upHeap(self.size)
end

-- Updates a value (and rebalances)
function BinaryHeap:update(i, priority, val)
  if i > self.size then
    print('Error: index ' .. i .. ' is greater than the current size of the heap')
    return
  end

  -- Replace value
  self.array[i][1] = priority
  self.array[i][2] = val
  -- Update hash table
  self.hash[val] = i

  -- Rebalance
  self:downHeap(i)
  self:upHeap(i)
end

-- Updates a value by using the value (using the hash table)
function BinaryHeap:updateByVal(valKey, priority, val)
  self:update(self.hash[valKey], priority, val)
end

-- Returns the maximum priority with value
function BinaryHeap:peek()
  return self.size ~= 0 and self.array[1] or nil
end

-- Removes and returns the maximum priority with value
function BinaryHeap:pop()
  -- Return nil if no values
  if self.size == 0 then
    print('Error: no values in heap')
    return nil
  end

  local max = self.array[1]:clone()

  -- Move the last value (not necessarily the smallest) to the root
  self.array[1] = self.array[self.size]
  self.size = self.size - 1
  -- Update hash table
  self.hash[self.array[1][2]] = 1

  -- Rebalance
  self:downHeap(1)

  return max
end

-- Rebalances the heap (by moving large values up)
function BinaryHeap:upHeap(i)
  -- Calculate parent index
  local p = math.floor(i/2)

  if i > 1 then
    -- If parent is smaller than child then swap
    if self.array[p][1] < self.array[i][1] then
      self.array[i], self.array[p] = self.array[p]:clone(), self.array[i]:clone()
      -- Update hash table
      self.hash[self.array[i][2]], self.hash[self.array[p][2]] = i, p

      -- Continue rebalancing
      self:upHeap(p)
    end
  end
end

-- Rebalances the heap (by moving small values down)
function BinaryHeap:downHeap(i)
  -- Calculate left and right child indices
  local l, r = 2*i, 2*i + 1

  -- Find the index of the greatest of these elements
  local greatest
  if l <= self.size and self.array[l][1] > self.array[i][1] then
    greatest = l
  else
    greatest = i
  end
  if r <= self.size and self.array[r][1] > self.array[greatest][1] then
    greatest = r
  end

  -- Continue rebalancing if necessary
  if greatest ~= i then
    self.array[i], self.array[greatest] = self.array[greatest]:clone(), self.array[i]:clone()
    -- Update hash table
    self.hash[self.array[i][2]], self.hash[self.array[greatest][2]] = i, greatest

    self:downHeap(greatest)
  end
end

-- Retrieves priorities
function BinaryHeap:getPriorities()
  return self.array:narrow(2, 1, 1)
end

-- Retrieves values
function BinaryHeap:getValues()
  return self.array:narrow(2, 2, 1)
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
      str = str .. '\n' .. string.rep('  ', math.pow(2, maxLevel - l))
      level = l
    end
    -- Print value and spacing
    str = str .. string.format('%.2f ', self.array[i][2]) .. string.rep('    ', maxLevel - l)
  end

  return str
end

-- Index using hash table
function BinaryHeap:__index(key)
  return self.array[self.hash[key]]
end

return BinaryHeap
