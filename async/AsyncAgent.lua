local AbstractAgent = require 'async/AbstractAgent'
local AsyncModel = require 'async/AsyncModel'
local CircularQueue = require 'structures/CircularQueue'
local classic = require 'classic'
local optim = require 'optim'
require 'modules/sharedRmsProp'

local AsyncAgent = classic.class('AsyncAgent', AbstractAgent)

local methods = {
  OneStepQ = 'OneStepQAgent',
  Sarsa = 'SarsaAgent',
  NStepQ = 'NStepQAgent',
  A3C = 'A3CAgent'
}

function AsyncAgent.static.build(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
    local Agent = require('async/'..methods[opt.async])
    return Agent(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
end


function AsyncAgent:_init(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  local asyncModel = AsyncModel(opt)
  self.env, self.model = asyncModel:getEnvAndModel()

  self.id = __threadid or 1
  self.atomic = atomic

  self.optimiser = optim[opt.optimiser]
  self.optimParams = {
    learningRate = opt.eta,
    momentum = opt.momentum,
    rmsEpsilon = opt.rmsEpsilon,
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

  self.stateBuffer = CircularQueue(opt.recurrent and 1 or opt.histLen, opt.Tensor, opt.stateSpec[2])

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


function AsyncAgent:applyGradients(net, dTheta, theta)
  if self.gradClip > 0 then
    net:gradParamClip(self.gradClip)
  end

  local feval = function()
    -- loss needed for validation stats only which is not computed for async yet, so just 0
    local loss = 0 -- 0.5 * tdErr ^2
    return loss, dTheta
  end

  self.optimParams.learningRate = self.learningRateStart * (self.totalSteps - self.step) / self.totalSteps
  self.optimiser(feval, theta, self.optimParams)

  dTheta:zero()
end


function AsyncAgent:observe()
  error('not implemented yet')
end


function AsyncAgent:training()
  error('not implemented yet')
end


function AsyncAgent:evaluate()
  error('not implemented yet')
end


return AsyncAgent
