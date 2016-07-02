local classic = require 'classic'
local QAgent = require 'async/AsyncAgent'

local QAgent, super = classic.class('QAgent', 'AsyncAgent')

local EPSILON_ENDS = { 0.01, 0.1, 0.5}
local EPSILON_PROBS = { 0.4, 0.7, 1 }


function QAgent:_init(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  super._init(self, opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  self.super = super

  self.targetNet = targetNet:clone('weight', 'bias')
  self.targetNet:evaluate()

  self.targetTheta = targetTheta
  local __, gradParams = self.policyNet:parameters()
  self.dTheta = nn.Module.flatten(gradParams)
  self.dTheta:zero()

  self.doubleQ = opt.doubleQ

  self.epsilonStart = opt.epsilonStart
  self.epsilon = self.epsilonStart
  self.PALpha = opt.PALpha

  self.target = self.Tensor(self.m)

  self.totalSteps = math.floor(opt.steps / opt.threads)

  self:setEpsilon(opt)
  self.tic = 0
  self.step = 0

  -- Forward state anyway if recurrent
  self.alwaysComputeGreedyQ = opt.recurrent or not self.doubleQ

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


function QAgent:eGreedy(state, net)
  self.epsilon = math.max(self.epsilonStart + (self.step - 1)*self.epsilonGrad, self.epsilonEnd)

  if self.alwaysComputeGreedyQ then
    self.QCurr = net:forward(state):squeeze()
  end

  if torch.uniform() < self.epsilon then
    return torch.random(1,self.m)
  end

  if not self.alwaysComputeGreedyQ then
    self.QCurr = net:forward(state):squeeze()
  end

  local _, maxIdx = self.QCurr:max(1)
  return maxIdx[1]
end


function QAgent:progress(steps)
  self.step = self.step + 1
  if self.atomic:inc() % self.tau == 0 then
    self.targetTheta:copy(self.theta)
    if self.tau>1000 then
      log.info('QAgent | updated targetNetwork at %d', self.atomic:get()) 
    end
  end
  if self.step % self.progFreq == 0 then
    local progressPercent = 100 * self.step / steps
    local speed = self.progFreq / torch.toc(self.tic)
    self.tic = torch.tic()
    log.info('AsyncAgent | step=%d | %.02f%% | speed=%d/sec | ε=%.2f -> %.2f | η=%.8f',
      self.step, progressPercent, speed ,self.epsilon, self.epsilonEnd, self.optimParams.learningRate)
  end
end


function QAgent:accumulateGradientTdErr(state, action, tdErr, net)
  if self.tdClip > 0 then
      if tdErr > self.tdClip then tdErr = self.tdClip end
      if tdErr <-self.tdClip then tdErr =-self.tdClip end
  end

  self.target:zero()
  self.target[action] = -tdErr

  net:backward(state, self.target)
end


return QAgent

