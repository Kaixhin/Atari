local _ = require 'moses'
local AsyncModel = require 'AsyncModel'
local Evaluator = require 'Evaluator'
local Experience = require 'Experience'
local CircularQueue = require 'structures/CircularQueue'
local classic = require 'classic'
local gnuplot = require 'gnuplot'
require 'classic.torch'

local ValidationAgent = classic.class('ValidationAgent')

function ValidationAgent:_init(opt, policyNet, counters)
  log.info('creating ValidationAgent')
  local asyncModel = AsyncModel(opt)
  self.env, self.model = asyncModel:getEnvAndModel()
  self.model:setNetwork(policyNet)

  self._id = opt._id

  self.counters = counters

  -- Validation variables
  self.valSize = opt.valSize
  self.losses = {}
  self.avgV = {} -- Running average of V(s')
  self.avgTdErr = {} -- Running average of TD-error δ
  self.valScores = {} -- Validation scores (passed from main script)
  self.normScores = {} -- Normalised validation scores (passed from main script)

  local actionSpec = self.env:getActionSpec()
  self.m = actionSpec[3][2] - actionSpec[3][1] + 1
  self.actionOffset = 1 - actionSpec[3][1]

  self.policyNet = policyNet:clone('weight', 'bias')

  self.ale = opt.ale

  if self.ale then self.env:training() end

  self.stateBuffer = CircularQueue(opt.histLen, opt.Tensor, {opt.nChannels, opt.height, opt.width})
  self.progFreq = opt.progFreq
  self.Tensor = opt.Tensor

  self.reportWeights = opt.reportWeights
  self.valSteps = opt.valSteps
  self.evaluator = Evaluator(opt.game)

  opt.batchSize = opt.valSize -- override in this thread ONLY
  self.valMemory = Experience(opt.valSize + 3, opt, true)

  self:fillValMemory()

  classic.strict(self)
end


function ValidationAgent:fillValMemory()
  local reward, rawObservation, terminal = 0, self.env:start(), false
  local action = 1
  for i=1,self.valSteps+1 do
    local observation = self.model:preprocess(rawObservation)
    self.valMemory:store(reward, observation, terminal, action)
    if not terminal then
      action = torch.random(1,self.m)
      reward, rawObservation, terminal = self.env:step(action - self.actionOffset)
    else
      reward, rawObservation, terminal = 0, self.env:start(), false
    end
  end
  log.info('ValidationAgent | ValMemory filled')
end


function ValidationAgent:eGreedy0(state, epsilon)
  if torch.uniform() < epsilon then
    return torch.random(1,self.m)
  end

  local Q = self.policyNet:forward(state):squeeze()
  local _, maxIdx = Q:max(1)
  return maxIdx[1]
end


function ValidationAgent:validate()
  self.stateBuffer:clear()
  if self.ale then self.env:evaluate() end

  local valStepStrFormat = '%0' .. (math.floor(math.log10(self.valSteps)) + 1) .. 'd'
  local epsilon = 0.001 -- Taken from tuned DDQN evaluation
  local valEpisode = 1
  local valEpisodeScore = 0
  local valTotalScore = 0

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
        local avgScore = valTotalScore/math.max(valEpisode - 1, 1)
        log.info('[VAL] Steps: ' .. string.format(valStepStrFormat, valStep) .. '/' .. self.valSteps .. ' | Episode ' .. valEpisode
          .. ' | Score: ' .. valEpisodeScore .. ' | TotScore: ' .. valTotalScore .. ' | AvgScore: %.2f', avgScore)
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

  log.info('Total Score: ' .. valTotalScore)
  valTotalScore = valTotalScore/math.max(valEpisode - 1, 1) -- Only average score for completed episodes in general
  log.info('Average Score: ' .. valTotalScore)
  self.valScores[#self.valScores + 1] = valTotalScore
  local normScore = self.evaluator:normaliseScore(valTotalScore)
  if normScore then
    log.info('Normalised Score: ' .. normScore)
    self.normScores[#self.normScores + 1] = normScore
  end

  self:visualiseFilters()

  local avgV = self:validationStats()
  log.info('Average V: ' .. avgV)

  if self.reportWeights then
    local reports = self:weightsReport()
    for r = 1, #reports do
      log.info(reports[r])
    end
  end
end

-- Saves network convolutional filters as images
function ValidationAgent:visualiseFilters()
  local filters = self.model:getFilters()

  for i, v in ipairs(filters) do
    image.save(paths.concat('experiments', self._id, 'conv_layer_' .. i .. '.png'), v)
  end
end

local pprintArr = function(memo, v)
  return memo .. ', ' .. v
end

-- Reports absolute network weights and gradients
function ValidationAgent:weightsReport()
  -- Collect layer with weights
  local weightLayers = self.policyNet:findModules('nn.SpatialConvolution')
  local fcLayers = self.policyNet:findModules('nn.Linear')
  weightLayers = _.append(weightLayers, fcLayers)
  
  -- Array of norms and maxima
  local wNorms = {}
  local wMaxima = {}
  local wGradNorms = {}
  local wGradMaxima = {}

  -- Collect statistics
  for l = 1, #weightLayers do
    local w = weightLayers[l].weight:clone():abs() -- Weights (absolute)
    wNorms[#wNorms + 1] = torch.mean(w) -- Weight norms:
    wMaxima[#wMaxima + 1] = torch.max(w) -- Weight max
    w = weightLayers[l].gradWeight:clone():abs() -- Weight gradients (absolute)
    wGradNorms[#wGradNorms + 1] = torch.mean(w) -- Weight grad norms:
    wGradMaxima[#wGradMaxima + 1] = torch.max(w) -- Weight grad max
  end

  -- Create report string table
  local reports = {
    'Weight norms: ' .. _.reduce(wNorms, pprintArr),
    'Weight max: ' .. _.reduce(wMaxima, pprintArr),
    'Weight gradient norms: ' .. _.reduce(wGradNorms, pprintArr),
    'Weight gradient max: ' .. _.reduce(wGradMaxima, pprintArr)
  }

  return reports
end


function ValidationAgent:validationStats()
  local indices = torch.linspace(2, self.valSize+1, self.valSize):long()
  local states, actions, rewards, transitions, terminals = self.valMemory:retrieve(indices)

  local QPrimes = self.policyNet:forward(transitions) -- in real learning targetNet but doesnt matter for validation
  local VPrime = torch.max(QPrimes, 3)
  local totalV = VPrime:sum()
  local avgV = totalV / self.valSize
  self.avgV[#self.avgV + 1] = avgV
  self:plotValidation()
  return avgV
end


function ValidationAgent:plotValidation()
  -- TODO: Reduce memory consumption for gnuplot
  -- Plot and save losses
  if #self.losses > 0 then
    local losses = torch.Tensor(self.losses)
    gnuplot.pngfigure(paths.concat('experiments', self._id, 'losses.png'))
    gnuplot.plot('Loss', torch.linspace(math.floor(self.learnStart/self.progFreq), math.floor(self.globals.step/self.progFreq), #self.losses), losses, '-')
    gnuplot.xlabel('Step (x' .. self.progFreq .. ')')
    gnuplot.ylabel('Loss')
    gnuplot.plotflush()
    torch.save(paths.concat('experiments', self._id, 'losses.t7'), losses)
  end
  -- Plot and save V
  local epochIndices = torch.linspace(1, #self.avgV, #self.avgV)
  local Vs = torch.Tensor(self.avgV)
  gnuplot.pngfigure(paths.concat('experiments', self._id, 'Vs.png'))
  gnuplot.plot('V', epochIndices, Vs, '-')
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('V')
  gnuplot.movelegend('left', 'top')
  gnuplot.plotflush()
  torch.save(paths.concat('experiments', self._id, 'V.t7'), Vs)
  -- Plot and save TD-error δ
  if #self.avgTdErr>0 then
    local TDErrors = torch.Tensor(self.avgTdErr)
    gnuplot.pngfigure(paths.concat('experiments', self._id, 'TDErrors.png'))
    gnuplot.plot('TD-Error', epochIndices, TDErrors, '-')
    gnuplot.xlabel('Epoch')
    gnuplot.ylabel('TD-Error')
    gnuplot.plotflush()
    torch.save(paths.concat('experiments', self._id, 'TDErrors.t7'), TDErrors)
  end
  -- Plot and save average score
  local scores = torch.Tensor(self.valScores)
  gnuplot.pngfigure(paths.concat('experiments', self._id, 'scores.png'))
  gnuplot.plot('Score', epochIndices, scores, '-')
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('Average Score')
  gnuplot.movelegend('left', 'top')
  gnuplot.plotflush()
  torch.save(paths.concat('experiments', self._id, 'scores.t7'), scores)
    -- Plot and save normalised score
  if #self.normScores > 0 then
    local normScores = torch.Tensor(self.normScores)
    gnuplot.pngfigure(paths.concat('experiments', self._id, 'normScores.png'))
    gnuplot.plot('Score', epochIndices, normScores, '-')
    gnuplot.xlabel('Epoch')
    gnuplot.ylabel('Normalised Score')
    gnuplot.movelegend('left', 'top')
    gnuplot.plotflush()
    torch.save(paths.concat('experiments', self._id, 'normScores.t7'), normScores)
  end
end


return ValidationAgent

