local AsyncModel = require 'AsyncModel'
local CircularQueue = require 'structures/CircularQueue'
local classic = require 'classic'
local optim = require 'optim'
require 'modules/sharedRmsProp'

local AsyncAgent = classic.class('AsyncAgent')


function AsyncAgent:_init(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  log.info('creating AsyncAgent')
  local asyncModel = AsyncModel(opt)
  self.env, self.model = asyncModel:getEnvAndModel()

  self.id = __threadid or 1
  self.atomic = atomic

  self.optimiser = optim[opt.optimiser]
  self.optimParams = {
    learningRate = opt.eta,
    momentum = opt.momentum,
    g = sharedG
  }

  self.learningRateStart = opt.eta

  local actionSpec = self.env:getActionSpec()
  self.m = actionSpec[3][2] - actionSpec[3][1] + 1
  self.actionOffset = 1 - actionSpec[3][1]

  self.policyNet = policyNet:clone('weight', 'bias')

  self.theta = theta
  local __, gradParams = self.policyNet:parameters()
  self.dTheta = nn.Module.flatten(gradParams)
  self.dTheta:zero()

  self.ale = opt.ale

  self.stateBuffer = CircularQueue(opt.histLen, opt.Tensor, {opt.nChannels, opt.height, opt.width})

  self.gamma = opt.gamma
  self.rewardClip = opt.rewardClip
  self.tdClip = opt.tdClip

  self.progFreq = opt.progFreq
  self.batchSize = opt.batchSize
  self.gradClip = opt.gradClip
  self.tau = opt.tau
  self.Tensor = opt.Tensor

  self.batchIdx = 0

  self.totalSteps = math.floor(opt.steps / opt.threads)

  self.tic = 0
  self.step = 0
end


function AsyncAgent:start()
  local reward, rawObservation, terminal = 0, self.env:start(), false
  local observation = self.model:preprocess(rawObservation)
  self.stateBuffer:push(observation)
  return reward, terminal, self.stateBuffer:readAll()
end


function AsyncAgent:takeAction(action)
  local reward, rawObservation, terminal = self.env:step(action - self.actionOffset)
  if self.rewardClip > 0 then
    reward = math.max(reward, -self.rewardClip)
    reward = math.min(reward, self.rewardClip)
  end

  local observation = self.model:preprocess(rawObservation)
  if terminal then
    self.stateBuffer:pushReset(observation)
  else
    self.stateBuffer:push(observation)
  end

  return reward, terminal, self.stateBuffer:readAll()
end


function AsyncAgent:progress(steps)
  if self.step % self.progFreq == 0 then
    local progressPercent = 100 * self.step / steps
    local speed = self.progFreq / torch.toc(self.tic)
    self.tic = torch.tic()
    log.info('AsyncAgent | step=%d | %.02f%% | speed=%d/sec | ε=%.2f -> %.2f | η=%.8f',
      self.step, progressPercent, speed ,self.epsilon, self.epsilonEnd, self.optimParams.learningRate)
  end
end


function AsyncAgent:applyGradients(net, dTheta, theta)
  if self.gradClip > 0 then
    net:gradParamClip(self.gradClip)
  end

  local feval = function()
    local loss = 0 -- torch.mean(self.tdErr:clone():pow(2):mul(0.5))
    return loss, dTheta
  end

  self.optimParams.learningRate = self.learningRateStart * (self.totalSteps - self.step) / self.totalSteps
  self.optimiser(feval, theta, self.optimParams)

  dTheta:zero()
end


return AsyncAgent

