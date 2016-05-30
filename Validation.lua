local classic = require 'classic'
local Evaluator = require 'Evaluator'
local gnuplot = require 'gnuplot'
local _ = require 'moses'

local Validation = classic.class('Validation')


function Validation:_init(opt, agent, env, display)
  self.opt = opt
  self.agent = agent
  self.env = env
  self.display = display

  -- Create (Atari normalised score) evaluator
  self.evaluator = Evaluator(opt.game)

  self.bestValScore = _.max(self.agent.valScores) or -math.huge -- Retrieve best validation score from agent if available

  classic.strict(self)
end


function Validation:validate()
  log.info('Validating')
  -- Set environment and agent to evaluation mode
  if self.opt.ale then self.env:evaluate() end
  self.agent:evaluate()

  -- Start new game
  local reward, state, terminal = 0, self.env:start(), false

  -- Validation variables
  local valEpisode = 1
  local valEpisodeScore = 0
  local valTotalScore = 0
  local valStepStrFormat = '%0' .. (math.floor(math.log10(self.opt.valSteps)) + 1) .. 'd' -- String format for padding step with zeros

  for valStep = 1, self.opt.valSteps do
    -- Observe and choose next action (index)
    local action = self.agent:observe(reward, state, terminal)
    if not terminal then
      -- Act on environment
      reward, state, terminal = self.env:step(action)
      -- Track score
      valEpisodeScore = valEpisodeScore + reward
    else
      -- Print score every 10 episodes
      if valEpisode % 10 == 0 then
        log.info('[VAL] Steps: ' .. string.format(valStepStrFormat, valStep) .. '/' .. self.opt.valSteps .. ' | Episode ' .. valEpisode .. ' | Score: ' .. valEpisodeScore)
      end

      -- Start a new episode
      valEpisode = valEpisode + 1
      reward, state, terminal = 0, self.env:start(), false
      valTotalScore = valTotalScore + valEpisodeScore -- Only add to total score at end of episode
      valEpisodeScore = reward -- Reset episode score
    end

    self.display:display(self.agent, state)
  end

  -- If no episodes completed then use score from incomplete episode
  if valEpisode == 1 then
    valTotalScore = valEpisodeScore
  end

  -- Print total and average score
  log.info('Total Score: ' .. valTotalScore)
  valTotalScore = valTotalScore/math.max(valEpisode - 1, 1) -- Only average score for completed episodes in general
  log.info('Average Score: ' .. valTotalScore)
  -- Pass to agent (for storage and plotting)
  self.agent.valScores[#self.agent.valScores + 1] = valTotalScore
  -- Calculate normalised score (if valid)
  local normScore = self.evaluator:normaliseScore(valTotalScore)
  if normScore then
    log.info('Normalised Score: ' .. normScore)
    self.agent.normScores[#agent.normScores + 1] = normScore
  end

  -- Visualise convolutional filters
  self.agent:visualiseFilters()

  -- Use transitions sampled for validation to test performance
  local avgV, avgTdErr = self.agent:validate()
  log.info('Average V: ' .. avgV)
  log.info('Average δ: ' .. avgTdErr)

  -- Save if best score achieved
  if valTotalScore > self.bestValScore then
    log.info('New best average score')
    self.bestValScore = valTotalScore

    log.info('Saving weights')
    self.agent:saveWeights(paths.concat(self.opt.experiments, self.opt._id, 'weights.t7'))
  end
end


function Validation:evaluate()
  log.info('Evaluation mode')
  -- Set environment and agent to evaluation mode
  if self.opt.ale then self.env:evaluate() end
  self.agent:evaluate()

  local reward, state, terminal = 0, self.env:start(), false

  -- Report episode score
  local episodeScore = reward

  -- Play one game (episode)
  local step = 1
  while not terminal do
    -- Observe and choose next action (index)
    action = self.agent:observe(reward, state, terminal)
    -- Act on environment
    reward, state, terminal = self.env:step(action)
    episodeScore = episodeScore + reward

    self.display:recordAndDisplay(self.agent, state, step)
    -- Increment evaluation step counter
    step = step + 1
  end
  log.info('Final Score: ' .. episodeScore)

  self.display:createVideo()
end


return Validation
