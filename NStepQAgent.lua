local _ = require 'moses'
local AsyncModel = require 'AsyncModel'
local CircularQueue = require 'structures/CircularQueue'
local classic = require 'classic'
local optim = require 'optim'
require 'modules/sharedRmsProp'
require 'classic.torch'

local NStepQAgent, super = classic.class('NStepQAgent', 'QAgent')

function NStepQAgent:_init(opt, policyNet, targetNet, theta, counters, sharedG)
  super._init(self, opt, policyNet, targetNet, theta, counters, sharedG)
  classic.strict(self)
end


function NStepQAgent:learn(steps)
  self.step = self.counters[self.id]
  self.policyNet:training()
  self.stateBuffer:clear()
  if self.ale then self.env:training() end

  log.info('NStepQAgent starting | steps=%d | ε=%.2f -> %.2f', steps, self.epsilon, self.epsilonEnd)
  local reward, rawObservation, terminal = 0, self.env:start(), false
  local observation = self.model:preprocess(rawObservation)

  self.stateBuffer:push(observation)
  local state = self.stateBuffer:readAll()

  local action, state_

  self.tic = torch.tic()
  for step1=1,steps do
    
    -- TODO

    self:progress(steps)
  end

  log.info('NStepQAgent ended learning steps=%d ε=%.4f', steps, self.epsilon)
end


return NStepQAgent

