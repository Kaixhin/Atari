local _ = require 'moses'
local AsyncModel = require 'AsyncModel'
local CircularQueue = require 'structures/CircularQueue'
local classic = require 'classic'
local optim = require 'optim'
require 'modules/sharedRmsProp'
require 'classic.torch'

local QAgent = classic.class('QAgent')

local EPSILON_ENDS = { 0.01, 0.1, 0.5}
local EPSILON_PROBS = { 0.4, 0.7, 1 }

function QAgent:_init(opt, policyNet, targetNet, theta, counters, sharedG)
  log.info('creating QAgent')
  local asyncModel = AsyncModel(opt)
  self.env, self.model = asyncModel:getEnvAndModel()

  self.id = __threadid or 1
  self.counters = counters

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
  self.targetNet = targetNet:clone('weight', 'bias')
  self.targetNet:evaluate()

  self.theta = theta
  local __, gradParams = self.policyNet:parameters()
  self.dTheta = nn.Module.flatten(gradParams)
  self.dTheta:zero()

  self.ale = opt.ale
  self.doubleQ = opt.doubleQ

  self.stateBuffer = CircularQueue(opt.histLen, opt.Tensor, {opt.nChannels, opt.height, opt.width})

  self.gamma = opt.gamma
  self.rewardClip = opt.rewardClip
  self.tdClip = opt.tdClip
  self.epsilonStart = opt.epsilonStart
  self.epsilon = self.epsilonStart
  self.PALpha = opt.PALpha

  self.progFreq = opt.progFreq
  self.batchSize = opt.batchSize
  self.gradClip = opt.gradClip

  self.Tensor = opt.Tensor

  self.batchIdx = 0
  self.target = self.Tensor(self.m)

  self.totalSteps = math.floor(opt.steps / opt.threads)

  self:setEpsilon(opt)
  self.tic = 0
  self.step = 0

  self.QCurr = torch.Tensor(0)
end


function QAgent:setEpsilon(opt)
  local r = torch.rand(1):squeeze()
  local e = 3
  if r < EPSILON_PROBS[1] then
    e = 1
  elseif r < EPSILON_PROBS[2] then
    e = 2
  end
  self.epsilonEnd = EPSILON_ENDS[e]
  self.epsilonGrad = (self.epsilonEnd - opt.epsilonStart) / opt.epsilonSteps
end



function QAgent:eGreedy(state)
  self.epsilon = math.max(self.epsilonStart + (self.step - 1)*self.epsilonGrad, self.epsilonEnd)

  self.QCurr = self.policyNet:forward(state):squeeze()

  if torch.uniform() < self.epsilon then
    return torch.random(1,self.m)
  end

  local _, maxIdx = self.QCurr:max(1)
  return maxIdx[1]
end


function QAgent:accumulateGradient(state, action, state_, reward, terminal)
  local Y = reward
  if not terminal then
      local QPrimes = self.targetNet:forward(state_):squeeze()
      local APrimeMax = QPrimes:max(1):squeeze()

      if self.doubleQ then
          local _,APrimeMaxInds = self.policyNet:forward(state_):squeeze():max(1)
          APrimeMax = QPrimes[APrimeMaxInds[1]]
      end

      Y = Y + self.gamma * APrimeMax
  end

  local tdErr = Y - self.QCurr[action]

  if self.tdClip > 0 then
      if tdErr > self.tdClip then tdErr = self.tdClip end
      if tdErr <-self.tdClip then tdErr =-self.tdClip end
  end

  self.target:zero()
  self.target[action] = -tdErr
  self.policyNet:backward(state, self.target)
end


function QAgent:applyGradients()
  if self.gradClip > 0 then
    self.policyNet:gradParamClip(self.gradClip)
  end

  local feval = function()
    local loss = 0 -- torch.mean(self.tdErr:clone():pow(2):mul(0.5))
    return loss, self.dTheta
  end

  self.optimParams.learningRate = self.learningRateStart * (self.totalSteps - self.step) / self.totalSteps

  self.optimiser(feval, self.theta, self.optimParams)
end


return QAgent

