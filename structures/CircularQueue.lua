local classic = require 'classic'
require 'classic.torch' -- Enables serialisation
require 'torchx'

-- A non-standard circular queue
local CircularQueue = classic.class('CircularQueue')

-- Creates a new fixed-length circular queue and tensor creation function
function CircularQueue:_init(length, createTensor, tensorSizes)
  self.length = length
  self.queue = {}
  self.reset = false

  -- Initialise zero tensors
  for i = 1, self.length do
    self.queue[#self.queue + 1] = createTensor(torch.LongStorage(tensorSizes)):zero()
  end
end

-- Pushes a new element to the end of the queue and moves all others down
function CircularQueue:push(tensor)
  if self.reset then
    -- If reset flag set, zero old tensors
    for i = 1, self.length - 1 do
      self.queue[i]:zero()
    end

    -- Unset reset flag
    self.reset = false
  else
    -- Otherwise, move old elements down
    for i = 1, self.length - 1 do
      self.queue[i] = self.queue[i + 1]
    end
  end

  -- Add new element (casting if needed, will keep reference if not)
  self.queue[self.length] = tensor:typeAs(self.queue[1])
end

-- Pushes a new element to the end of the queue and sets reset flag
function CircularQueue:pushReset(tensor)
  -- Move old elements down
  for i = 1, self.length - 1 do
    self.queue[i] = self.queue[i + 1]
  end

  -- Add new element (casting if needed, will keep reference if not)
  self.queue[self.length] = tensor:typeAs(self.queue[1])

  -- Set reset flag
  self.reset = true
end

-- Resets (zeros) the entire queue
function CircularQueue:clear()
  for i = 1, self.length do
    self.queue[i]:zero()
  end
end

-- Reads entire queue as a large tensor
function CircularQueue:readAll()
  return torch.concat(self.queue)
end

return CircularQueue
