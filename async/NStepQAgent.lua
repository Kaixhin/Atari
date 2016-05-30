local classic = require 'classic'
local optim = require 'optim'
local QAgent = require 'async/QAgent'
require 'modules/sharedRmsProp'

local NStepQAgent, super = classic.class('NStepQAgent', 'QAgent')


function NStepQAgent:_init(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  super._init(self, opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  self.policyNet_ = self.policyNet:clone()
  self.policyNet_:training()
  self.theta_, self.dTheta_ = self.policyNet_:getParameters()
  self.dTheta_:zero()

  self.rewards = torch.Tensor(self.batchSize)
  self.actions = torch.ByteTensor(self.batchSize)
  self.states = torch.Tensor(0)

  if self.ale then self.env:training() end

  self.alwaysComputeGreedyQ = false

  classic.strict(self)
end


function NStepQAgent:learn(steps, from)
  self.step = from or 0
  self.stateBuffer:clear()

  log.info('NStepQAgent starting | steps=%d | ε=%.2f -> %.2f', steps, self.epsilon, self.epsilonEnd)
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

      reward, terminal, state = self:takeAction(action)
      self.rewards[self.batchIdx] = reward

      self:progress(steps)
    until terminal or self.batchIdx == self.batchSize

    self:accumulateGradients(terminal, state)

    if terminal then 
      reward, terminal, state = self:start()
    end

    self:applyGradients(self.policyNet_, self.dTheta_, self.theta)
  until self.step >= steps

  log.info('NStepQAgent ended learning steps=%d ε=%.4f', steps, self.epsilon)
end


function NStepQAgent:accumulateGradients(terminal, state)
  local R = 0
  if not terminal then
    local QPrimes = self.targetNet:forward(state):squeeze()
    local APrimeMax = QPrimes:max(1):squeeze()

    if self.doubleQ then
        local _,APrimeMaxInds = self.policyNet_:forward(state):squeeze():max(1)
        APrimeMax = QPrimes[APrimeMaxInds[1]]
    end
    R = APrimeMax
  end

  for i=self.batchIdx,1,-1 do
    R = self.rewards[i] + self.gamma * R
    local Q_i = self.policyNet_:forward(self.states[i]):squeeze()
    local tdErr = R - Q_i[self.actions[i]]
    self:accumulateGradientTdErr(self.states[i], self.actions[i], tdErr, self.policyNet_) 
  end
end


return NStepQAgent
