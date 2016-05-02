local AsyncModel = require 'AsyncModel'
local CircularQueue = require 'structures/CircularQueue'
local classic = require 'classic'
local optim = require 'optim'
require 'modules/rmspropm' -- Add RMSProp with momentum
require 'classic.torch'

local AsyncAgent = classic.class('AsyncAgent')


function AsyncAgent:_init(opt, policyNet, targetNet, theta, counters)
  log.info('creating AsyncAgent')
  local asyncModel = AsyncModel(opt)
  self.env, self.model = asyncModel:getEnvAndModel()

  self.id = __threadid or 1
  self.counters = counters

  self.optimiser = optim[opt.optimiser]
  self.optimParams = {
    learningRate = opt.eta,
    momentum = opt.momentum
  }

  local actionSpec = self.env:getActionSpec()
  self.m = actionSpec[3][2] - actionSpec[3][1] + 1
  self.actionOffset = 1 - actionSpec[3][1]

  self.policyNet = policyNet:clone('weight', 'bias')
  self.targetNet = targetNet:clone('weight', 'bias')
  self.targetNet:evaluate()

  self.theta = theta
  local __, gradParams = self.policyNet:parameters()
  self.dTheta = nn.Module.flatten(gradParams)
  self.dTheta:zero()

  self.ale = opt.ale
  self.tau = opt.tau
  self.doubleQ = opt.doubleQ

  self.stateBuffer = CircularQueue(opt.histLen, opt.Tensor, {opt.nChannels, opt.height, opt.width})

  self.gamma = opt.gamma
  self.rewardClip = opt.rewardClip
  self.tdClip = opt.tdClip
  self.epsilonStart = opt.epsilonStart
  self.epsilonEnd = opt.epsilonEnd
  self.epsilonGrad = (opt.epsilonEnd - opt.epsilonStart)/opt.epsilonSteps
  self.epsilon = self.epsilonStart
  self.PALpha = opt.PALpha

  self.batchSize = opt.batchSize
  self.gradClip = opt.gradClip

  self.Tensor = opt.Tensor

  self.batchIdx = 0
  self.target = self.Tensor(self.m)

  self.step = 0
  self.valSteps = opt.valSteps
  classic.strict(self)
end


function AsyncAgent:learn(steps)
  self.policyNet:training()
  self.stateBuffer:clear()
  if self.ale then self.env:training() end

  log.info('AsyncAgent starting learning steps=%d ε=%.4f', steps, self.epsilon)
  local reward, rawObservation, terminal = 0, self.env:start(), false
  local observation = self.model:preprocess(rawObservation)

  self.stateBuffer:push(observation)
  local state = self.stateBuffer:readAll()

  local action, state_

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
      reward, observation, terminal = 0, self.env:start(), false
      rawObservation = self.model:preprocess(observation)
      self.stateBuffer:push(rawObservation)
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
  end
end


function AsyncAgent:eGreedy(state)
  self.epsilon = math.max(self.epsilonStart + (self.step - 1)*self.epsilonGrad, self.epsilonEnd)
  return self:eGreedy0(state, self.epsilon)
end


function AsyncAgent:eGreedy0(state, epsilon)
  if torch.uniform() < epsilon then
    return torch.random(1,self.m)
  end

  local Q = self.policyNet:forward(state):squeeze()
  local _, maxIdx = Q:max(1)
  return maxIdx[1]
end


function AsyncAgent:accumulateGradient(state, action, state_, reward, terminal)
  local Y = reward
  if not terminal then
      local q2s = self.targetNet:forward(state_):squeeze()
      local q2 = q2s:max(1):squeeze()

      if self.doubleQ then
          local _,argmax = self.policyNet:forward(state_):squeeze():max(1)
          q2 = q2s[argmax[1]]
      end

      Y = Y + self.gamma * q2
  end

  local qs = self.policyNet:forward(state):squeeze()
  local tdErr = Y - qs[action]

  if self.tdClip > 0 then
      if tdErr > self.tdClip then tdErr = self.tdClip end
      if tdErr <-self.tdClip then tdErr =-self.tdClip end
  end

  self.target:zero()
  self.target[action] = -tdErr
  self.policyNet:backward(state, self.target)
end


function AsyncAgent:applyGradients()
  if self.gradClip > 0 then
    self.policyNet:gradParamClip(self.gradClip)
  end

  local feval = function()
    local loss = 0 -- torch.mean(self.tdErr:clone():pow(2):mul(0.5))
    return loss, self.dTheta
  end

  self.optimiser(feval, self.theta, self.optimParams)
end


function AsyncAgent:validate()
  self.stateBuffer:clear()
  if self.ale then self.env:evaluate() end

  local valStepStrFormat = '%0' .. (math.floor(math.log10(self.valSteps)) + 1) .. 'd'
  local epsilon = 0.001 -- Taken from tuned DDQN evaluation
  local valEpisode = 1
  local valEpisodeScore = 0
  local valTotalScore = 0
  local normScore = 0

  local reward, observation, terminal = 0, self.env:start(), false

  for valStep = 1, self.valSteps do
    observation = self.model:preprocess(observation)
    if terminal then
      self.stateBuffer:clear()
    else
      self.stateBuffer:push(observation)
    end
    if not terminal then
      local state = self.stateBuffer:readAll()

      local action = self:eGreedy0(state, epsilon)
      reward, observation, terminal = self.env:step(action - self.actionOffset)
      valEpisodeScore = valEpisodeScore + reward
    else
      -- Print score every 10 episodes
      if valEpisode % 10 == 0 then
        log.info('[VAL] Steps: ' .. string.format(valStepStrFormat, valStep) .. '/' .. self.valSteps .. ' | Episode ' .. valEpisode .. ' | Score: ' .. valEpisodeScore)
      end

      -- Start a new episode
      valEpisode = valEpisode + 1
      reward, observation, terminal = 0, self.env:start(), false
      valTotalScore = valTotalScore + valEpisodeScore -- Only add to total score at end of episode
      valEpisodeScore = reward -- Reset episode score
    end
  end

  -- If no episodes completed then use score from incomplete episode
  if valEpisode == 1 then
    valTotalScore = valEpisodeScore
  end

  -- Print total and average score
  log.info('Total Score: ' .. valTotalScore)
  valTotalScore = valTotalScore/math.max(valEpisode - 1, 1) -- Only average score for completed episodes in general
  log.info('Average Score: ' .. valTotalScore)
end


return AsyncAgent

