local classic = require 'classic'
local optim = require 'optim'
require 'modules/sharedRmsProp'

local A3CAgent,super = classic.class('A3CAgent', 'AsyncAgent')


function A3CAgent:_init(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  super._init(self, opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)

  log.info('creating A3CAgent')

  self.policyNet_ = policyNet:clone('weight', 'bias')

  __, self.dTheta_ = self.policyNet_:getParams()
  self.dTheta_:zero()

  self.target = self.Tensor(self.m)

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


