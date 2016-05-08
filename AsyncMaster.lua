require 'socket'
local AsyncModel = require 'AsyncModel'
local OneStepQAgent = require 'OneStepQAgent'
local ValidationAgent = require 'ValidationAgent'
local class = require 'classic'
local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')

local AsyncMaster = classic.class('AsyncMaster')

local TARGET_UPDATER = 1
local VALIDATOR = 2

local function checkNotNan(t)
  local sum = t:sum()
  local ok = sum == sum
  if not ok then
    log.error('ERROR'.. sum)
  end
  assert(ok)
end

local function torchSetup(opt)
  local tensorType = opt.tensorType
  local seed = opt.seed
  return function()
    log.info('Setting up Torch7')
    require 'nn'
    -- Use enhanced garbage collector
    torch.setheaptracking(true)
    -- Set number of BLAS threads
    -- must be 1 for each thread
    torch.setnumthreads(1)
    -- Set default Tensor type (float is more efficient than double)
    torch.setdefaulttensortype(tensorType)
    -- Set manual seed: but different for each thread
    -- to have different experiences, eg. catch randomness
    torch.manualSeed(seed * __threadid)
  end
end  

local function threadedFormatter(thread)
  local threadName = thread

  return function(level, ...)
    local msg = nil

    if #{...} > 1 then
        msg = string.format(({...})[1], unpack(fn.rest({...})))
    else
        msg = pprint.pretty_string(({...})[1])
    end

    return string.format("[%s: %s - %s] - %s\n", threadName, logroll.levels[level], os.date("%Y_%m_%d_%X"), msg)
  end
end

local function setupLogging(opt, thread)
  local _id = opt._id
  local threadName = thread
  return function()
    require 'logroll'
    local thread = threadName or __threadid
    if type(thread) == 'number' then
      thread = ('%02d'):format(thread)
    end
    local file = paths.concat('experiments', _id, 'log.'.. thread ..'.txt')
    local flog = logroll.file_logger(file)
    local formatterFunc = threadedFormatter(thread)
    local plog = logroll.print_logger({formatter = formatterFunc})
    log = logroll.combine(flog, plog)
  end
end


function AsyncMaster:_init(opt)
  self.opt = opt

  -- not atomic, but calling sum() on it is good enough
  self.counters = torch.LongTensor(opt.threads)

  local asyncModel = AsyncModel(opt)

  local policyNet = asyncModel:createNet()
  local targetNet = policyNet:clone()
  local counters = self.counters

  self.theta = policyNet:getParameters()
  self.targetTheta = targetNet:getParameters()
  local sharedG = self.theta:clone():zero()

  self.controlPool = threads.Threads(2)
  self.controlPool:specific(true)

  self.controlPool:addjob(TARGET_UPDATER, setupLogging(opt, 'TG'))
  self.controlPool:addjob(TARGET_UPDATER, torchSetup(opt))

  self.controlPool:addjob(VALIDATOR, setupLogging(opt, 'VA'))
  self.controlPool:addjob(VALIDATOR, torchSetup(opt))
  self.controlPool:addjob(VALIDATOR, function()
    local ValidationAgent = require 'ValidationAgent'
    validAgent = ValidationAgent(opt, policyNet, counters)
  end)

  self.controlPool:synchronize()

  local theta = self.theta

  -- without locking xitari sometimes crashes during initialization
  -- but not later... but is it really threadsafe then...?
  local mutex = threads.Mutex()
  local mutexId = mutex:id()
  self.pool = threads.Threads(self.opt.threads, function()
    end,
    setupLogging(opt),
    torchSetup(opt),
    function()
      local threads1 = require 'threads'
      local mutex1 = threads1.Mutex(mutexId)
      mutex1:lock()
      local OneStepQAgent = require 'OneStepQAgent'
      agent = OneStepQAgent(opt, policyNet, targetNet, theta, counters, sharedG)
      mutex1:unlock()
    end
  )
  mutex:free()

  classic.strict(self)
end


function AsyncMaster:start()
  local counters = self.counters

  local opt = self.opt
  local theta = self.theta
  local targetTheta = self.targetTheta

  local validator = function()
    require 'socket'
    validAgent:start()
    local lastUpdate = 0
    while true do
      local countSum = counters:sum()
      if countSum < 0 then return end

      local countSince = countSum - lastUpdate
      if countSince > opt.valFreq then
        log.info('starting validation after %d steps', countSince)
        lastUpdate = countSum
        validAgent:validate()
      end
      socket.select(nil,nil,1)
    end
  end

  self.controlPool:addjob(VALIDATOR, validator)

  local targetUpdater = function()
    require 'socket'
    local lastUpdate = 0
    local sleepSecs  = 0.01
    if opt.tau > 1000 then sleepSecs = 1 end
    while true do
      local countSum = counters:sum()
      if countSum < 0 then return end

      local countSince = countSum - lastUpdate
      if countSince > opt.tau then
        lastUpdate = countSum
        targetTheta:copy(theta)
        checkNotNan(targetTheta)
        if opt.tau > 1000 then
          log.info('updated targetNet from policyNet after %d steps', countSince)
        end
      end
      socket.select(nil,nil,sleepSecs)
    end
  end

  self.controlPool:addjob(TARGET_UPDATER, targetUpdater)

  local steps1 = self.opt.steps / self.opt.threads

  for i=1,self.opt.threads do
    self.pool:addjob(function()
      agent:learn(steps1)
    end)
  end

  self.pool:synchronize()
  counters:fill(-1)

  self.controlPool:synchronize()

  self.pool:terminate()
  self.controlPool:terminate()
end

return AsyncMaster

