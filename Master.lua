local _ = require 'moses'
local classic = require 'classic'
local signal = require 'posix.signal'
local gnuplot = require 'gnuplot'
local Singleton = require 'structures/Singleton'
local Agent = require 'Agent'
local Evaluator = require 'Evaluator'
local Display = require 'Display'

local Master = classic.class('Master')

function Master:_init(opt)
  self.opt = opt

  -- Set up singleton global object for transferring step
  self.globals = Singleton({step = 1}) -- Initial step

  -- Initialise Catch or Arcade Learning Environment
  log.info('Setting up ' .. (opt.ale and 'Arcade Learning Environment' or 'Catch'))
  local Env = opt.ale and require 'rlenvs.Atari' or require 'rlenvs.Catch'
  self.env = Env(opt)
  local stateSpec = self.env:getStateSpec()
  -- Provide original channels, height and width for resizing from
  opt.origChannels, opt.origHeight, opt.origWidth = table.unpack(stateSpec[2])

  -- Create DQN agent
  log.info('Creating DQN')
  self.agent = Agent(self.env, opt)
  if paths.filep(opt.network) then
    -- Load saved agent if specified
    log.info('Loading pretrained network weights')
    self.agent:loadWeights(opt.network)
  elseif paths.filep(paths.concat(opt.experiments, opt._id, 'agent.t7')) then
    -- Ask to load saved agent if found in experiment folder (resuming training)
    log.info('Saved agent found - load (y/n)?')
    if io.read() == 'y' then
      log.info('Loading saved agent')
      self.agent = torch.load(paths.concat(opt.experiments, opt._id, 'agent.t7'))

      -- Reset globals (step) from agent
      Singleton.setInstance(self.agent.globals)
      self.globals = Singleton.getInstance()

      -- Switch saliency style
      self.agent:setSaliency(opt.saliency)
    end
  end

  self.bestValScore = _.max(self.agent.valScores) or -math.huge -- Retrieve best validation score from agent if available

  -- Create (Atari normalised score) evaluator
  self.evaluator = Evaluator(opt.game)

  -- Start gaming
  log.info('Starting game: ' .. opt.game)
  local state = self.env:start()
  self.display = Display(opt, state)

  classic.strict(self)
end


function Master:train()
  self:catchSigInt()

  local reward, state, terminal = 0, self.env:start(), false

  log.info('Training mode')

  -- Set environment and agent to training mode
  if self.opt.ale then self.env:training() end
  self.agent:training()

  -- Training variables (reported in verbose mode)
  local episode = 1
  local episodeScore = reward

  -- Training loop
  local initStep = self.globals.step -- Extract step
  local stepStrFormat = '%0' .. (math.floor(math.log10(self.opt.steps)) + 1) .. 'd' -- String format for padding step with zeros
  for step = initStep, self.opt.steps do
    self.globals.step = step -- Pass step number to globals for use in other modules
    
    -- Observe results of previous transition (r, s', terminal') and choose next action (index)
    local action = self.agent:observe(reward, state, terminal) -- As results received, learn in training mode
    if not terminal then
      -- Act on environment (to cause transition)
      reward, state, terminal = self.env:step(action)
      -- Track score
      episodeScore = episodeScore + reward
    else
      if self.opt.verbose then
        -- Print score for episode
        log.info('Steps: ' .. string.format(stepStrFormat, step) .. '/' .. self.opt.steps .. ' | Episode ' .. episode .. ' | Score: ' .. episodeScore)
      end

      -- Start a new episode
      episode = episode + 1
      reward, state, terminal = 0, self.env:start(), false
      episodeScore = reward -- Reset episode score
    end

    self.display:display(self.agent, state)

    -- Trigger learning after a while (wait to accumulate experience)
    if step == self.opt.learnStart then
      log.info('Learning started')
    end

    -- Report progress
    if step % self.opt.progFreq == 0 then
      log.info('Steps: ' .. string.format(stepStrFormat, step) .. '/' .. self.opt.steps)
      -- Report weight and weight gradient statistics
      if self.opt.reportWeights then
        local reports = self.agent:report()
        for r = 1, #reports do
          log.info(reports[r])
        end
      end
    end

    -- Validate
    if step >= self.opt.learnStart and step % self.opt.valFreq == 0 then
      self:validate()

      -- Start new game (as previous one was interrupted)
      reward, state, terminal = 0, self.env:start(), false
      episodeScore = reward
    end
  end

  log.info('Finished training')
end


function Master:validate()
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

  log.info('Resuming training')
  -- Set environment and agent to training mode
  if self.opt.ale then self.env:training() end
  self.agent:training()
end


function Master:evaluate()
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


-- Set up SIGINT (Ctrl+C) handler to save network before quitting
function Master:catchSigInt()
  signal.signal(signal.SIGINT, function(signum)
    log.warn('SIGINT received')
    log.info('Save agent (y/n)?')
    if io.read() == 'y' then
      log.info('Saving agent')
      torch.save(paths.concat(self.opt.experiments, self.opt._id, 'agent.t7'), self.agent) -- Save agent to resume training
    end
    log.warn('Exiting')
    os.exit(128 + signum)
  end)
end


return Master
