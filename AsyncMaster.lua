local AsyncModel = require 'AsyncModel'
local AsyncAgent = require 'AsyncAgent'
local class = require 'classic'
local threads = require 'threads'
threads.Threads.serialization('threads.sharedserialize')

local AsyncMaster = classic.class('AsyncMaster')

local function checkNotNan(t)
  local ok = t:ne(t):sum() == 0
  if not ok then
    log.error('ERROR'.. t:sum())
  end
  assert(ok)
end

local function torchSetup(opt)
  local tensorType = opt.tensorType
  local seed = opt.seed
  return function()
    log.info('Setting up Torch7')
    -- Use enhanced garbage collector
    torch.setheaptracking(true)
    -- Set number of BLAS threads
    torch.setnumthreads(1)
    -- Set default Tensor type (float is more efficient than double)
    torch.setdefaulttensortype(tensorType)
    -- Set manual seed
    torch.manualSeed(seed)
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
  self.counters = torch.LongTensor(opt.threads)

  local asyncModel = AsyncModel(opt)

  local policyNet = asyncModel:createNet()
  local targetNet = policyNet:clone()
  local counters = self.counters

  self.theta = policyNet:getParameters()
  self.targetTheta = targetNet:getParameters()

  self.controlPool = threads.Threads(1,
    setupLogging(opt, 'TG'),
    torchSetup(opt))

  local theta = self.theta

  -- without locking xitari sometimes crashes during initialization
  -- but not later... but is it really threadsafe then...?
  local mutex = threads.Mutex()
  local mutexId = mutex:id()

  self.pool = threads.Threads(self.opt.threads, function()
      require 'nn'
      AsyncAgent1 = require 'AsyncAgent'
    end,
    setupLogging(opt),
    torchSetup(opt),
    function()
      local threads1 = require 'threads'
      local mutex1 = threads1.Mutex(mutexId)
      mutex1:lock()
      agent = AsyncAgent1(opt, policyNet, targetNet, theta, counters)
      mutex1:unlock()
    end
  )

  mutex:free()

  self.evalAgent = AsyncAgent(opt, policyNet, targetNet, theta, counters)

  classic.strict(self)
end


function AsyncMaster:start()
  local counters = self.counters

  local opt = self.opt
  local theta = self.theta
  local targetTheta = self.targetTheta

  local targetUpdater = function()
    require 'socket'
    local lastUpdate = 0
    while true do
      local countSum = counters:sum()
      if countSum < 0 then return end

      local countSince = countSum - lastUpdate
      if countSince > opt.tau then
        lastUpdate = countSum
        targetTheta:copy(theta)
        checkNotNan(targetTheta)
--        log.info('updated targetNet from policyNet after %d steps', countSince)
      end
      socket.select(nil,nil,.01)
    end
  end

  self.controlPool:addjob(targetUpdater)

  while true do

log.info('theta=%f', self.theta:sum())
log.info('targetTheta=%f', self.targetTheta:sum())

    for i=1,self.opt.threads do
      self.pool:addjob(function()
        agent:learn(opt.valFreq)
      end)
    end

    self.pool:synchronize()

    self.evalAgent:validate()
  end

  counter:fill(-1)
  self.pool:terminate()
  self.pool2:terminate()
end

return AsyncMaster

