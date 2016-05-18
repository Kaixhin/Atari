local classic = require 'classic'
local optim = require 'optim'
require 'modules/sharedRmsProp'

local A3CAgent,super = classic.class('A3CAgent', 'AsyncAgent')


function A3CAgent:_init(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  super._init(self, opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)

  log.info('creating A3CAgent')

  self.policyNet_ = policyNet:clone('weight', 'bias')

  self.theta_, self.dTheta_ = self.policyNet_:getParameters()
  self.dTheta_:zero()

  self.policyTarget = self.Tensor(self.m)
  self.vTarget = self.Tensor(1)
  self.targets = { vTarget, policyTarget }

  self.rewards = torch.Tensor(self.batchSize)
  self.actions = torch.ByteTensor(self.batchSize)
  self.states = torch.Tensor(0)

  self.beta = 0.01

  classic.strict(self)
end


function A3CAgent:learn(steps, from)
  self.step = from or 0

  self.stateBuffer:clear()
  if self.ale then self.env:training() end

  log.info('A3CAgent starting | steps=%d', steps)
  local reward, terminal, state = self:start()

  self.states:resize(self.batchSize, unpack(state:size():totable()))

  self.tic = torch.tic()

  repeat
    self.theta_:copy(self.theta)
    self.batchIdx = 0
    repeat
      self.batchIdx = self.batchIdx + 1
      self.states[self.batchIdx]:copy(state)

      if V == nil then
        V, probability = unpack(self.policyNet_:forward(state))
      end
      local action = torch.multinomial(probability, 1):squeeze()

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
  until self.step == steps

  log.info('A3CAgent ended learning steps=%d', steps)
end


function A3CAgent:accumulateGradients(terminal, state)
  local R = 0
  local V, probability
  if not terminal then
    V, probability = unpack(self.policyNet_:forward(state))
    R = V
  end

  for i=self.batchIdx,1-1 do
    R = self.rewards[i] + self.gamma * R
    
    local action = self.actions[i]
    local V, probability = unpack(self.policyNet_:forward(self.states[i]))

    local advantage = R - V

    self.vTarget = - advantage
    self.policyTarget:zero()
    self.policyTarget[action] = advantage / probability[action] - self.beta * (probability:log():sum()+1)

    self.policyNet_:backward(self.targets)
  end

end


function A3CAgent:progress(steps)
  if self.atomic:inc() % self.progFreq == 0 then
    local progressPercent = 100 * self.step / steps
    local speed = self.progFreq / torch.toc(self.tic)
    self.tic = torch.tic()
    log.info('A3CAgent | step=%d | %.02f%% | speed=%d/sec | η=%.8f',
      self.step, progressPercent, speed, self.optimParams.learningRate)
  end
end

return A3CAgent

