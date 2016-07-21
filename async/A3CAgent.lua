local classic = require 'classic'
local optim = require 'optim'
local AsyncAgent = require 'async/AsyncAgent'
require 'modules/sharedRmsProp'

local A3CAgent,super = classic.class('A3CAgent', 'AsyncAgent')

local TINY_EPSILON = 1e-20

function A3CAgent:_init(opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)
  super._init(self, opt, policyNet, targetNet, theta, targetTheta, atomic, sharedG)

  log.info('creating A3CAgent')

  self.policyNet_ = policyNet:clone()

  self.theta_, self.dTheta_ = self.policyNet_:getParameters()
  self.dTheta_:zero()

  self.policyTarget = self.Tensor(self.m)
  self.vTarget = self.Tensor(1)
  self.targets = { self.vTarget, self.policyTarget }

  self.rewards = torch.Tensor(self.batchSize)
  self.actions = torch.ByteTensor(self.batchSize)
  self.states = torch.Tensor(0)
  self.beta = opt.entropyBeta

  self.env:training()

  classic.strict(self)
end


function A3CAgent:learn(steps, from)
  self.step = from or 0

  self.stateBuffer:clear()

  log.info('A3CAgent starting | steps=%d', steps)
  local reward, terminal, state = self:start()

  self.states:resize(self.batchSize, table.unpack(state:size():totable()))

  self.tic = torch.tic()
  repeat
    self.theta_:copy(self.theta)
    self.batchIdx = 0
    repeat
      self.batchIdx = self.batchIdx + 1
      self.states[self.batchIdx]:copy(state)

      local V, probability = table.unpack(self.policyNet_:forward(state))
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
  until self.step >= steps

  log.info('A3CAgent ended learning steps=%d', steps)
end


function A3CAgent:accumulateGradients(terminal, state)
  local R = 0
  if not terminal then
    R = self.policyNet_:forward(state)[1]
  end

  for i=self.batchIdx,1,-1 do
    R = self.rewards[i] + self.gamma * R

    local action = self.actions[i]
    local V, probability = table.unpack(self.policyNet_:forward(self.states[i]))
    probability:add(TINY_EPSILON) -- could contain 0 -> log(0)= -inf -> theta = nans

    self.vTarget[1] = -0.5 * (R - V)

    -- ∇θ logp(s) = 1/p(a) for chosen a, 0 otherwise
    self.policyTarget:zero()
    -- f(s) ∇θ logp(s)
    self.policyTarget[action] = -(R - V) / probability[action] -- Negative target for gradient descent

    -- Calculate (negative of) gradient of entropy of policy (for gradient descent): -(-logp(s) - 1)
    local gradEntropy = torch.log(probability) + 1
    -- Add to target to improve exploration (prevent convergence to suboptimal deterministic policy)
    self.policyTarget:add(self.beta, gradEntropy)
    
    self.policyNet_:backward(self.states[i], self.targets)
  end
end


function A3CAgent:progress(steps)
  self.atomic:inc()
  self.step = self.step + 1
  if self.step % self.progFreq == 0 then
    local progressPercent = 100 * self.step / steps
    local speed = self.progFreq / torch.toc(self.tic)
    self.tic = torch.tic()
    log.info('A3CAgent | step=%d | %.02f%% | speed=%d/sec | η=%.8f',
      self.step, progressPercent, speed, self.optimParams.learningRate)
  end
end

return A3CAgent
