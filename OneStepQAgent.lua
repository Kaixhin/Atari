local _ = require 'moses'
local AsyncModel = require 'AsyncModel'
local CircularQueue = require 'structures/CircularQueue'
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
  local reward, rawObservation, terminal = 0, self.env:start(), false
  local observation = self.model:preprocess(rawObservation)

  self.stateBuffer:push(observation)
  local state = self.stateBuffer:readAll()

  local action, state_

  self.tic = torch.tic()
  for step1=1,steps do
    if not terminal then
      action = self:eGreedy(state)
      reward, rawObservation, terminal = self.env:step(action - self.actionOffset)
      observation = self.model:preprocess(rawObservation)

      if terminal then
        self.stateBuffer:pushReset(observation)
      else
        self.stateBuffer:push(observation)
      end

      if self.rewardClip > 0 then
        reward = math.max(reward, -self.rewardClip)
        reward = math.min(reward, self.rewardClip)
      end
    else
      reward, rawObservation, terminal = 0, self.env:start(), false
      observation = self.model:preprocess(rawObservation)
      self.stateBuffer:push(observation)
    end
    state_ = self.stateBuffer:readAll()

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
      self.dTheta:zero()
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

  if self.tdClip > 0 then
      if tdErr > self.tdClip then tdErr = self.tdClip end
      if tdErr <-self.tdClip then tdErr =-self.tdClip end
  end

  self.target:zero()
  self.target[action] = -tdErr
  self.policyNet:backward(state, self.target)
end

return OneStepQAgent

