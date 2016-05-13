local _ = require 'moses'
local classic = require 'classic'
local optim = require 'optim'
require 'modules/sharedRmsProp'
require 'classic.torch'


local OneStepQAgent, super = classic.class('OneStepQAgent', 'QAgent')


function OneStepQAgent:_init(opt, policyNet, targetNet, theta, counters, sharedG)
  super._init(self, opt, policyNet, targetNet, theta, counters, sharedG)
  classic.strict(self)
end


function OneStepQAgent:learn(steps)
  self.step = self.counters[self.id]
  self.policyNet:training()
  self.stateBuffer:clear()
  if self.ale then self.env:training() end

  log.info('OneStepQAgent starting | steps=%d | ε=%.2f -> %.2f', steps, self.epsilon, self.epsilonEnd)
  local reward, terminal, state = self:start()

  local action, state_

  self.tic = torch.tic()
  for step1=1,steps do
    if not terminal then
      action = self:eGreedy(state, self.policyNet)
      reward, terminal, state_ = self:takeAction(action)
    else
      reward, terminal, state_ = self:start()
    end

    if state ~= nil then
      self:accumulateGradient(state, action, state_, reward, terminal)
      self.batchIdx = self.batchIdx + 1
    end

    if not terminal then
      state = state_
    else
      state = nil
    end

    if self.batchIdx == self.batchSize or terminal then
      self:applyGradients()
      self.batchIdx = 0
    end

    self:progress(steps)
  end

  log.info('OneStepQAgent ended learning steps=%d ε=%.4f', steps, self.epsilon)
end


function OneStepQAgent:accumulateGradient(state, action, state_, reward, terminal)
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

  self:accumulateGradientTdErr(state, action, tdErr)
end

return OneStepQAgent

