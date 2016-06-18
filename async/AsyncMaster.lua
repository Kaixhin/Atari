local classic = require 'classic'
local threads = require 'threads'
local tds = require 'tds'
local signal = require 'posix.signal'
local AsyncModel = require 'async/AsyncModel'
local AsyncAgent = require 'async/AsyncAgent'
local QAgent = require 'async/QAgent'
local OneStepQAgent = require 'async/OneStepQAgent'
local NStepQAgent = require 'async/NStepQAgent'
local A3CAgent = require 'async/A3CAgent'
local ValidationAgent = require 'async/ValidationAgent'
require 'socket'
threads.Threads.serialization('threads.sharedserialize')

local FINISHED = -99999999

local AsyncMaster = classic.class('AsyncMaster')


local function checkNotNan(t)
  local sum = t:sum()
  local ok = sum == sum
  if not ok then
    log.error('ERROR '.. sum)
  end
  assert(ok)
end

local function torchSetup(opt)
  local tensorType = opt.tensorType
  local seed = opt.seed
  return function()
    log.info('Setting up Torch7')
    require 'nn'
    require 'rnn'
    require 'nngraph'
    require 'modules/GradientRescale'
    -- Use enhanced garbage collector
    torch.setheaptracking(true)
    -- Set number of BLAS threads to 1 (per thread)
    torch.setnumthreads(1)
    -- Set default Tensor type (float is more efficient than double)
    torch.setdefaulttensortype(tensorType)
    -- Set manual seed (different for each thread to have different experiences)
    torch.manualSeed(seed * __threadid)
  end
end  

local function threadedFormatter(thread)
  local threadName = thread

  return function(level, ...)
    local msg = nil

    if #{...} > 1 then
        msg = string.format(({...})[1], table.unpack(fn.rest({...})))
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
    unpack = table.unpack -- TODO: Remove global unpack from dependencies
    -- Create log10 for Lua 5.2
    if not math.log10 then
      math.log10 = function(x)
        return math.log(x, 10)
      end
    end

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

  self.stateFile = paths.concat('experiments', opt._id, 'agent.async.t7')

  local asyncModel = AsyncModel(opt)
  local policyNet = asyncModel:createNet()
  self.theta = policyNet:getParameters()

  log.info('%s', policyNet)

  if paths.filep(opt.network) then
    log.info('Loading pretrained network weights')
    local weights = torch.load(opt.network)
    self.theta:copy(weights)
  end

  self.atomic = tds.AtomicCounter()

  local targetNet = policyNet:clone()
  self.targetTheta = targetNet:getParameters()
  local sharedG = self.theta:clone():zero()

  local theta = self.theta
  local targetTheta = self.targetTheta
  local stateFile = self.stateFile
  local atomic = self.atomic

  self.controlPool = threads.Threads(1)

  self.controlPool:addjob(setupLogging(opt, 'VA'))
  self.controlPool:addjob(torchSetup(opt))
  self.controlPool:addjob(function()
    local signal = require 'posix.signal'
    local ValidationAgent = require 'async/ValidationAgent'
    validAgent = ValidationAgent(opt, theta, atomic)
    if not opt.noValidation then
      signal.signal(signal.SIGINT, function(signum)
        log.warn('SIGINT received')
        log.info('Saving agent')
        local globalSteps = atomic:get()
        local state = { globalSteps = globalSteps }
        torch.save(stateFile, state)

        validAgent:saveWeights('last')
        log.warn('Exiting')
        os.exit(128 + signum)
      end)
    end
  end)

  self.controlPool:synchronize()

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
      local AsyncAgent = require 'async/AsyncAgent'
      local mutex1 = threads1.Mutex(mutexId)
      mutex1:lock()
      agent = AsyncAgent.build(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
      mutex1:unlock()
    end
  )
  mutex:free()

  classic.strict(self)
end


function AsyncMaster:start()
  local stepsToGo = math.floor(self.opt.steps / self.opt.threads)
  local startStep = 0
  if self.opt.network ~= '' and paths.filep(self.stateFile) then
      local state = torch.load(self.stateFile)
      stepsToGo = math.floor((self.opt.steps - state.globalSteps) / self.opt.threads)
      startStep = math.floor(state.globalSteps / self.opt.threads)
      self.atomic:set(state.globalSteps)
      log.info('Resuming training from step %d', state.globalSteps)
      log.info('Loading pretrained network weights')
      local weights = torch.load(paths.concat('experiments', self.opt._id, 'last.weights.t7'))
      self.theta:copy(weights)
      self.targetTheta:copy(self.theta)
  end

  local atomic = self.atomic
  local opt = self.opt
  local theta = self.theta
  local targetTheta = self.targetTheta

  local validator = function()
    local posix = require 'posix'
    validAgent:start()
    local lastUpdate = 0
    while true do
      local globalStep = atomic:get()
      if globalStep < 0 then return end

      local countSince = globalStep - lastUpdate
      if countSince > opt.valFreq then
        log.info('starting validation after %d steps', countSince)
        lastUpdate = globalStep
        local status, err = xpcall(validAgent.validate, debug.traceback, validAgent)
        if not status then
          log.error('%s', err)
          os.exit(128)
        end
      end
      posix.sleep(1)
    end
  end

  if not self.opt.noValidation then
    self.controlPool:addjob(validator)
  end

  for i=1,self.opt.threads do
    self.pool:addjob(function()
      local status, err = xpcall(agent.learn, debug.traceback, agent, stepsToGo, startStep)
      if not status then
        log.error('%s', err)
        os.exit(128)
      end
    end)
  end

  self.pool:synchronize()
  self.atomic:set(FINISHED)

  self.controlPool:synchronize()

  self.pool:terminate()
  self.controlPool:terminate()
end

return AsyncMaster

