local classic = require 'classic'
local optim = require 'optim'
require 'modules/sharedRmsProp'

local A3CAgent,super = classic.class('A3CAgent', 'AsyncAgent')


function A3CAgent:_init(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  super._init(self, opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  
  log.info('creating QAgent')
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

  self.policyNet_ = policyNet:clone('weight', 'bias')

  self.theta = theta
  __, self.dTheta = self.policyNet_:getParams()
  self.dTheta:zero()

  self.ale = opt.ale

  self.stateBuffer = CircularQueue(opt.histLen, opt.Tensor, {opt.nChannels, opt.height, opt.width})

  self.gamma = opt.gamma
  self.rewardClip = opt.rewardClip
  self.tdClip = opt.tdClip
  self.gradClip = opt.gradClip

  self.progFreq = opt.progFreq
  self.t_max = opt.batchSize

  self.Tensor = opt.Tensor

  self.T = 0
  self.target = self.Tensor(self.m)

  self.totalSteps = math.floor(opt.steps / opt.threads)

  self.tic = 0
  self.step = 0

  classic.strict(self)
end


function A3CAgent:learn(steps, from)
  self.step = from or 0

  self.stateBuffer:clear()
  if self.ale then self.env:training() end

  log.info('A3CAgent starting | steps=%d | ε=%.2f -> %.2f', steps, self.epsilon, self.epsilonEnd)
  local reward, terminal, state = self:start()

  self.states:resize(self.batchSize, unpack(state:size():totable()))

  self.tic = torch.tic()
  repeat
    self.theta_:copy(self.theta)
    self.batchIdx = 0
    repeat
      self.batchIdx = self.batchIdx + 1
      self.states[self.batchIdx]:copy(state)

      local action = self:eGreedy(state, self.policyNet_)
      self.actions[self.batchIdx] = action
      self.Qs[self.batchIdx]:copy(self.QCurr)

      reward, terminal, state = self:takeAction(action)
      self.rewards[self.batchIdx] = reward

      self:progress(steps)
    until terminal or self.batchIdx == self.batchSize

    self:accumulateGradients(terminal, state)

    if terminal then 
      reward, terminal, state = self:start()
    end

    self:applyGradients(self.policyNet_, self.dTheta_, self.theta)
  until self.step == steps

  log.info('A3CAgent ended learning steps=%d ε=%.4f', steps, self.epsilon)
end


