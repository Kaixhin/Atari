local classic = require 'classic'
local QAgent = require 'async/OneStepQAgent'

local SarsaAgent, super = classic.class('SarsaAgent', 'OneStepQAgent')


function SarsaAgent:_init(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  super._init(self, opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  log.info('creating SarsaAgent')
  self.agentName = 'SarsaAgent'
  classic.strict(self)
end


function SarsaAgent:accumulateGradient(state, action, state_, reward, terminal)
  local Y = reward
  local Q_state = self.QCurr[action]

  if not terminal then
      local action_ = self:eGreedy(state_, self.targetNet)

      Y = Y + self.gamma * self.QCurr[action_]
  end

  local tdErr = Y - Q_state

  self:accumulateGradientTdErr(state, action, tdErr, self.policyNet)
end


return SarsaAgent
