local _ = require 'moses'
local AsyncModel = require 'AsyncModel'
local CircularQueue = require 'structures/CircularQueue'
local classic = require 'classic'
local optim = require 'optim'
require 'modules/sharedRmsProp'
require 'classic.torch'

local OneStepQAgent, super = classic.class('OneStepQAgent', 'QAgent')

local EPSILON_ENDS = { 0.01, 0.1, 0.5}
local EPSILON_PROBS = { 0.4, 0.7, 1 }

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

    self.step = self.step + 1
    self.counters[self.id] = self.counters[self.id] + 1
    if self.step % self.progFreq == 0 then
      local progressPercent = 100 * self.step / steps
      local speed = self.progFreq / torch.toc(self.tic)
      self.tic = torch.tic()
      log.info('OneStepQAgent | step=%d | %.02f%% | speed=%d/sec | ε=%.2f -> %.2f | η=%.8f',
        self.step, progressPercent, speed ,self.epsilon, self.epsilonEnd, self.optimParams.learningRate)
    end
  end

  log.info('OneStepQAgent ended learning steps=%d ε=%.4f', steps, self.epsilon)
end


return OneStepQAgent

