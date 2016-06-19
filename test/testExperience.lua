local Singleton = require 'structures/Singleton'
local Experience = require 'Experience'

local Test = torch.TestSuite()
local standalone = tester == nil
if standalone then
  tester = torch.Tester()
end

torch.manualSeed(1)

local globals = Singleton({step = 1})

local isValidation = false
local capacity = 1e4
local opt = {
  histLen = 1,
  stateSpec = {
    'real',
    {1, 10, 10, 10},
    {0, 1}
  },
  discretiseMem = true,
  batchSize = 10,
  gpu = false,
  memPriority = '',
  learnStart = 0,
  steps = 1e6,
  alpha = .65,
  betaZero = 0.45,
  Tensor = torch.Tensor,
}


local function randomPopulate(priorities, experience) 
  local state = torch.Tensor(table.unpack(opt.stateSpec[2]))
  local terminal = false
  local action = 1
  local reward = 1
  local idx = torch.Tensor(1)
  local prio = torch.Tensor(1)
  local maxPrio = 1000

  for i=1,capacity do
    experience:store(reward, state, terminal, action)
    idx[1] = i
    prio[1] = torch.random(maxPrio) - maxPrio / 2
    priorities[i] = prio[1]
    experience:updatePriorities(idx, prio)
  end
end

local function samplePriorityMeans(times)
  local experience = Experience(capacity, opt, isValidation)
  local priorities = torch.Tensor(capacity)
  randomPopulate(priorities, experience)

  local samplePriorities = torch.Tensor(times, opt.batchSize)

  for i=1,times do
    local idxs = experience:sample()
    samplePriorities[i] = priorities:gather(1, idxs)
  end

--    print(samplePriorities)
  local means = samplePriorities:abs():mean(1):squeeze()
  print(means)

  return means
end

function Test:TestExperience_TestUniform()
  torch.manualSeed(1)
  opt.memPriority = false
  local means = samplePriorityMeans(1000)

  for i=1,means:size(1) do
    tester:assert(means[i]>235 and means[i]<265)
  end
end

function Test:TestExperience_TestRank()
  torch.manualSeed(1)
  opt.memPriority = 'rank'
  local means = samplePriorityMeans(10)

  for i=2,means:size(1) do
    tester:assertle(means[i], means[i-1])
  end
end


if standalone then
  tester:add(Test)
  tester:run()
end

return Test
