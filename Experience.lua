local _ = require 'moses'
local classic = require 'classic'
local BinaryHeap = require 'structures/BinaryHeap'
local Singleton = require 'structures/Singleton'
require 'classic.torch' -- Enables serialisation

local Experience = classic.class('Experience')

-- Creates experience replay memory
function Experience:_init(capacity, opt, isValidation)
  self.capacity = capacity
  -- Extract relevant options
  self.batchSize = opt.batchSize
  self.histLen = opt.histLen
  self.gpu = opt.gpu
  self.discretiseMem = opt.discretiseMem
  self.memPriority = opt.memPriority
  self.learnStart = opt.learnStart
  self.alpha = opt.alpha
  self.betaZero = opt.betaZero

  -- Create transition tuples buffer
  local bufferStateSize = torch.LongStorage(_.append({opt.batchSize, opt.histLen}, opt.stateSpec[2]))
  self.transTuples = {
    states = opt.Tensor(bufferStateSize),
    actions = torch.ByteTensor(opt.batchSize),
    rewards = opt.Tensor(opt.batchSize),
    transitions = opt.Tensor(bufferStateSize),
    terminals = torch.ByteTensor(opt.batchSize),
    priorities = opt.Tensor(opt.batchSize)
  }
  self.indices = torch.LongTensor(opt.batchSize)
  self.w = opt.Tensor(opt.batchSize):fill(1) -- Importance-sampling weights w, 1 if no correction needed

  -- Allocate memory for experience
  local stateSize = torch.LongStorage(_.append({capacity}, opt.stateSpec[2])) -- Calculate state storage size
  self.imgDiscLevels = 255 -- Number of discretisation levels for images (used for float <-> byte conversion)
  if opt.discretiseMem then
    -- For the standard DQN problem, float vs. byte storage is 24GB vs. 6GB memory, so this prevents/minimises slow swap usage
    self.states = torch.ByteTensor(stateSize) -- ByteTensor to avoid massive memory usage
  else
    self.states = torch.Tensor(stateSize)
  end
  self.actions = torch.ByteTensor(capacity) -- Discrete action indices
  self.rewards = torch.FloatTensor(capacity) -- Stored at time t (not t + 1)
  -- Terminal conditions stored at time t+1, encoded by 0 = false, 1 = true
  self.terminals = torch.ByteTensor(capacity):fill(1) -- Filling with 1 prevents going back in history at beginning
  -- Validation flags (used if state is stored without transition)
  self.invalid = torch.ByteTensor(capacity) -- 1 is used to denote invalid
  -- Internal pointer
  self.index = 1
  self.isFull = false
  self.size = 0

  -- TD-error δ-based priorities
  self.priorityQueue = BinaryHeap(capacity) -- Stored at time t
  self.smallConst = 1e-12
  -- Sampling priority
  if not isValidation and opt.memPriority == 'rank' then
    -- Cache partition indices for several values of N as α is static
    self.distributions = {}
    local nPartitions = 100 -- learnStart must be at least 1/100 of capacity (arbitrary constant)
    local partitionNum = 1
    local partitionDivision = math.floor(capacity/nPartitions)

    for n = partitionDivision, capacity, partitionDivision do
      if n >= opt.learnStart or n == capacity then -- Do not calculate distributions for before learnStart occurs
        -- Set up power-law PDF and CDF
        local distribution = {}
        distribution.pdf = torch.linspace(1, n, n):pow(-opt.alpha)
        local pdfSum = torch.sum(distribution.pdf)
        distribution.pdf:div(pdfSum) -- Normalise PDF
        local cdf = torch.cumsum(distribution.pdf)

        -- Set up strata for stratified sampling (transitions will have varying TD-error magnitudes |δ|)
        distribution.strataEnds = torch.LongTensor(opt.batchSize + 1)
        distribution.strataEnds[1] = 0 -- First index is 0 (+1)
        distribution.strataEnds[opt.batchSize + 1] = n -- Last index is n
        -- Use linear search to find strata indices
        local stratumEnd = 1/opt.batchSize
        local index = 1
        for s = 2, opt.batchSize do
          while cdf[index] < stratumEnd do
            index = index + 1
          end
          distribution.strataEnds[s] = index -- Save index
          stratumEnd = stratumEnd + 1/opt.batchSize -- Set condition for next stratum
        end

        -- Check that enough transitions are available (to prevent an infinite loop of infinite tuples)
        if distribution.strataEnds[2] - distribution.strataEnds[1] <= opt.histLen then
          log.error('Experience replay strata are too small - use a smaller alpha/larger memSize/greater learnStart')
          error('Experience replay strata are too small - use a smaller alpha/larger memSize/greater learnStart')
        end

        -- Store distribution
        self.distributions[partitionNum] = distribution
      end

      partitionNum = partitionNum + 1
    end
  end

  -- Initialise first time step (s0)
  self.states[1]:zero() -- Blank out state
  self.terminals[1] = 0
  self.actions[1] = 1 -- Action is no-op
  self.invalid[1] = 0 -- First step is a fake blanked-out state, but can thereby be utilised
  if self.memPriority then
    self.priorityQueue:insert(1, 1) -- First priority = 1
  end

  -- Calculate β growth factor (linearly annealed till end of training)
  self.betaGrad = (1 - opt.betaZero)/(opt.steps - opt.learnStart)

  -- Get singleton instance for step
  self.globals = Singleton.getInstance()
end

-- Calculates circular indices
function Experience:circIndex(x)
  local ind = x % self.capacity
  return ind == 0 and self.capacity or ind -- Correct 0-index
end

-- Stores experience tuple parts (including pre-emptive action)
function Experience:store(reward, state, terminal, action)
  self.rewards[self.index] = reward

  -- Increment index and size
  self.index = self.index + 1
  self.size = math.min(self.size + 1, self.capacity)
  -- Circle back to beginning if memory limit reached
  if self.index > self.capacity then
    self.isFull = true -- Full memory flag
    self.index = 1 -- Reset index
  end

  if self.discretiseMem then
    self.states[self.index] = torch.mul(state, self.imgDiscLevels) -- float -> byte
  else
    self.states[self.index] = state:clone()
  end
  self.terminals[self.index] = terminal and 1 or 0
  self.actions[self.index] = action
  self.invalid[self.index] = 0

  -- Store with maximal priority
  if self.memPriority then
    -- TODO: Correct PER by not storing terminal states at all
    local maxPriority = terminal and 0 or self.priorityQueue:findMax() -- Terminal states cannot be sampled so assign priority 0
    if self.isFull then
      self.priorityQueue:updateByVal(self.index, maxPriority, self.index)
    else
      self.priorityQueue:insert(maxPriority, self.index)
    end
  end
end

-- Sets current state as invalid (utilised when switching to evaluation mode)
function Experience:setInvalid()
  self.invalid[self.index] = 1
end

-- Retrieves experience tuples (s, a, r, s', t)
function Experience:retrieve(indices)
  local N = indices:size(1)
  -- Blank out history in one go
  self.transTuples.states:zero()
  self.transTuples.transitions:zero()

  -- Iterate over indices
  for n = 1, N do
    local memIndex = indices[n]
    -- Retrieve action
    self.transTuples.actions[n] = self.actions[memIndex]
    -- Retrieve rewards
    self.transTuples.rewards[n] = self.rewards[memIndex]
    -- Retrieve terminal status (of transition)
    self.transTuples.terminals[n] = self.terminals[self:circIndex(memIndex + 1)]

    -- Go back in history whilst episode exists
    local histIndex = self.histLen
    repeat
      if self.discretiseMem then
        -- Copy state (converting to float first for non-integer division)
        self.transTuples.states[n][histIndex]:div(self.states[memIndex]:typeAs(self.transTuples.states), self.imgDiscLevels) -- byte -> float
      else
        self.transTuples.states[n][histIndex] = self.states[memIndex]:typeAs(self.transTuples.states)
      end
      -- Adjust indices
      memIndex = self:circIndex(memIndex - 1)
      histIndex = histIndex - 1
    until histIndex == 0 or self.terminals[memIndex] == 1 or self.invalid[memIndex] == 1

    -- If transition not terminal, fill in transition history (invalid states should not be selected in the first place)
    if self.transTuples.terminals[n] == 0 then
      -- Copy most recent state
      for h = 2, self.histLen do
        self.transTuples.transitions[n][h - 1] = self.transTuples.states[n][h]
      end
      -- Get transition frame
      local memTIndex = self:circIndex(indices[n] + 1)
      if self.discretiseMem then
        self.transTuples.transitions[n][self.histLen]:div(self.states[memTIndex]:typeAs(self.transTuples.transitions), self.imgDiscLevels) -- byte -> float
      else
        self.transTuples.transitions[n][self.histLen] = self.states[memTIndex]:typeAs(self.transTuples.transitions)
      end
    end
  end

  return self.transTuples.states[{{1, N}}], self.transTuples.actions[{{1, N}}], self.transTuples.rewards[{{1, N}}], self.transTuples.transitions[{{1, N}}], self.transTuples.terminals[{{1, N}}]
end

-- Determines if an index points to a valid transition state
function Experience:validateTransition(index)
  -- Calculate beginning of state and end of transition for checking overlap with head of buffer
  local minIndex, maxIndex = index - self.histLen, self:circIndex(index + 1)
  -- State must not be terminal, invalid, or overlap with head of buffer
  return self.terminals[index] == 0 and self.invalid[index] == 0 and (self.index <= minIndex or self.index >= maxIndex)
end

-- Returns indices and importance-sampling weights based on (stochastic) proportional prioritised sampling
function Experience:sample()
  local N = self.size

  -- Priority 'none' = uniform sampling
  if not self.memPriority then

    -- Keep uniformly picking random indices until indices filled
    for n = 1, self.batchSize do
      local index
      local isValid = false

      -- Generate random index until valid transition found
      while not isValid do
        index = torch.random(1, N)
        isValid = self:validateTransition(index)
      end

      -- Store index
      self.indices[n] = index
    end

  elseif self.memPriority == 'rank' then

    -- Find closest precomputed distribution by size
    local distIndex = math.floor(N / self.capacity * 100)
    local distribution = self.distributions[distIndex]
    N = distIndex * 100

    -- Create table to store indices (by rank)
    local rankIndices = torch.LongTensor(self.batchSize) -- In reality the underlying array-based binary heap is used as an approximation of a ranked (sorted) array
    -- Perform stratified sampling
    for n = 1, self.batchSize do
      local index
      local isValid = false

      -- Generate random index until valid transition found
      while not isValid do
        -- Sample within stratum
        rankIndices[n] = torch.random(distribution.strataEnds[n] + 1, distribution.strataEnds[n+1])
        -- Retrieve actual transition index
        index = self.priorityQueue:getValueByVal(rankIndices[n])
        isValid = self:validateTransition(index) -- The last stratum might be full of terminal states, leading to many checks
      end

      -- Store actual transition index
      self.indices[n] = index
    end

    -- Compute importance-sampling weights w = (N * p(rank))^-β
    local beta = math.min(self.betaZero + (self.globals.step - self.learnStart - 1)*self.betaGrad, 1)
    self.w = distribution.pdf:index(1, rankIndices):mul(N):pow(-beta) -- torch.index does memory copy
    -- Calculate max importance-sampling weight
    -- Note from Tom Schaul: Calculated over minibatch, not entire distribution
    local wMax = torch.max(self.w)
    -- Normalise weights so updates only scale downwards (for stability)
    self.w:div(wMax)

  elseif self.memPriority == 'proportional' then

    -- TODO: Proportional prioritised experience replay

  end

  return self.indices, self.w
end

-- Update experience priorities using TD-errors δ
function Experience:updatePriorities(indices, delta)
  if self.memPriority then
    local priorities = torch.abs(delta):float() -- Use absolute values
    if self.memPriority == 'proportional' then
      priorities:add(self.smallConstant) -- Allows transitions to be sampled even if error is 0
    end

    for p = 1, indices:size(1) do
      self.priorityQueue:updateByVal(indices[p], priorities[p], indices[p]) 
    end
  end
end

-- Rebalance prioritised experience replay heap
function Experience:rebalance()
  self.priorityQueue:rebalance()
end

return Experience
