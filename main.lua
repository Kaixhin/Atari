local signal = require 'posix.signal'
local _ = require 'moses'
local gnuplot = require 'gnuplot'
local Singleton = require 'structures/Singleton'
local Agent = require 'Agent'
local Evaluator = require 'Evaluator'
local Setup = require 'Setup'
local Display = require 'Display'

local setup = Setup(arg)
local opt = setup.opt

-- Set up singleton global object for transferring step
local globals = Singleton({step = 1}) -- Initial step


----- Environment + Agent Setup -----

-- Initialise Catch or Arcade Learning Environment
log.info('Setting up ' .. (opt.ale and 'Arcade Learning Environment' or 'Catch'))
local env, stateSpec
if opt.ale then
  local Atari = require 'rlenvs.Atari'
  env = Atari(opt)
  stateSpec = env:getStateSpec()

  -- Provide original channels, height and width for resizing from
  opt.origChannels, opt.origHeight, opt.origWidth = table.unpack(stateSpec[2])
else
  local Catch = require 'rlenvs.Catch'
  env = Catch()
  stateSpec = env:getStateSpec()
  
  -- Provide original channels, height and width for resizing from
  opt.origChannels, opt.origHeight, opt.origWidth = table.unpack(stateSpec[2])

  -- Adjust height and width
  opt.height, opt.width = stateSpec[2][2], stateSpec[2][3]
end


-- Create DQN agent
log.info('Creating DQN')
local agent = Agent(env, opt)
if paths.filep(opt.network) then
  -- Load saved agent if specified
  log.info('Loading pretrained network weights')
  agent:loadWeights(opt.network)
elseif paths.filep(paths.concat(opt.experiments, opt._id, 'agent.t7')) then
  -- Ask to load saved agent if found in experiment folder (resuming training)
  log.info('Saved agent found - load (y/n)?')
  if io.read() == 'y' then
    log.info('Loading saved agent')
    agent = torch.load(paths.concat(opt.experiments, opt._id, 'agent.t7'))

    -- Reset globals (step) from agent
    Singleton.setInstance(agent.globals)
    globals = Singleton.getInstance()

    -- Switch saliency style
    agent:setSaliency(opt.saliency)
  end
end

----- Training / Evaluation -----

-- Create (Atari normalised score) evaluator
local evaluator = Evaluator(opt.game)

-- Start gaming
log.info('Starting game: ' .. opt.game)
local reward, state, terminal = 0, env:start(), false
local action

local display = Display(opt, state)

if opt.mode == 'train' then

  log.info('Training mode')

  -- Set up SIGINT (Ctrl+C) handler to save network before quitting
  signal.signal(signal.SIGINT, function(signum)
    log.warn('SIGINT received')
    log.info('Save agent (y/n)?')
    if io.read() == 'y' then
      log.info('Saving agent')
      torch.save(paths.concat(opt.experiments, opt._id, 'agent.t7'), agent) -- Save agent to resume training
    end
    log.warn('Exiting')
    os.exit(128 + signum)
  end)

  -- Set environment and agent to training mode
  if opt.ale then env:training() end
  agent:training()

  -- Training variables (reported in verbose mode)
  local episode = 1
  local episodeScore = reward

  -- Validation variables
  local valEpisode, valEpisodeScore, valTotalScore, normScore
  local bestValScore = _.max(agent.valScores) or -math.huge -- Retrieve best validation score from agent if available
  local valStepStrFormat = '%0' .. (math.floor(math.log10(opt.valSteps)) + 1) .. 'd' -- String format for padding step with zeros

  -- Training loop
  local initStep = globals.step -- Extract step
  local stepStrFormat = '%0' .. (math.floor(math.log10(opt.steps)) + 1) .. 'd' -- String format for padding step with zeros
  for step = initStep, opt.steps do
    globals.step = step -- Pass step number to globals for use in other modules
    
    -- Observe results of previous transition (r, s', terminal') and choose next action (index)
    action = agent:observe(reward, state, terminal) -- As results received, learn in training mode
    if not terminal then
      -- Act on environment (to cause transition)
      reward, state, terminal = env:step(action)
      -- Track score
      episodeScore = episodeScore + reward
    else
      if opt.verbose then
        -- Print score for episode
        log.info('Steps: ' .. string.format(stepStrFormat, step) .. '/' .. opt.steps .. ' | Episode ' .. episode .. ' | Score: ' .. episodeScore)
      end

      -- Start a new episode
      episode = episode + 1
      reward, state, terminal = 0, env:start(), false
      episodeScore = reward -- Reset episode score
    end

    display:display(agent, state)

    -- Trigger learning after a while (wait to accumulate experience)
    if step == opt.learnStart then
      log.info('Learning started')
    end

    -- Report progress
    if step % opt.progFreq == 0 then
      log.info('Steps: ' .. string.format(stepStrFormat, step) .. '/' .. opt.steps)
      -- Report weight and weight gradient statistics
      if opt.reportWeights then
        local reports = agent:report()
        for r = 1, #reports do
          log.info(reports[r])
        end
      end
    end

    -- Validate
    if step >= opt.learnStart and step % opt.valFreq == 0 then
      log.info('Validating')
      -- Set environment and agent to evaluation mode
      if opt.ale then env:evaluate() end
      agent:evaluate()

      -- Start new game
      reward, state, terminal = 0, env:start(), false

      -- Reset validation variables
      valEpisode = 1
      valEpisodeScore = 0
      valTotalScore = 0
      normScore = 0

      for valStep = 1, opt.valSteps do
        -- Observe and choose next action (index)
        action = agent:observe(reward, state, terminal)
        if not terminal then
          -- Act on environment
          reward, state, terminal = env:step(action)
          -- Track score
          valEpisodeScore = valEpisodeScore + reward
        else
          -- Print score every 10 episodes
          if valEpisode % 10 == 0 then
            log.info('[VAL] Steps: ' .. string.format(valStepStrFormat, valStep) .. '/' .. opt.valSteps .. ' | Episode ' .. valEpisode .. ' | Score: ' .. valEpisodeScore)
          end

          -- Start a new episode
          valEpisode = valEpisode + 1
          reward, state, terminal = 0, env:start(), false
          valTotalScore = valTotalScore + valEpisodeScore -- Only add to total score at end of episode
          valEpisodeScore = reward -- Reset episode score
        end

        display:display(agent, state)
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
      agent.valScores[#agent.valScores + 1] = valTotalScore
      -- Calculate normalised score (if valid)
      normScore = evaluator:normaliseScore(valTotalScore)
      if normScore then
        log.info('Normalised Score: ' .. normScore)
        agent.normScores[#agent.normScores + 1] = normScore
      end

      -- Visualise convolutional filters
      agent:visualiseFilters()

      -- Use transitions sampled for validation to test performance
      local avgV, avgTdErr = agent:validate()
      log.info('Average V: ' .. avgV)
      log.info('Average δ: ' .. avgTdErr)

      -- Save if best score achieved
      if valTotalScore > bestValScore then
        log.info('New best average score')
        bestValScore = valTotalScore

        log.info('Saving weights')
        agent:saveWeights(paths.concat(opt.experiments, opt._id, 'weights.t7'))
      end

      log.info('Resuming training')
      -- Set environment and agent to training mode
      if opt.ale then env:training() end
      agent:training()

      -- Start new game (as previous one was interrupted)
      reward, state, terminal = 0, env:start(), false
      episodeScore = reward
    end
  end

  log.info('Finished training')

elseif opt.mode == 'eval' then

  log.info('Evaluation mode')
  -- Set environment and agent to evaluation mode
  if opt.ale then env:evaluate() end
  agent:evaluate()

  -- Report episode score
  local episodeScore = reward

  -- Play one game (episode)
  local step = 1
  while not terminal do
    -- Observe and choose next action (index)
    action = agent:observe(reward, state, terminal)
    -- Act on environment
    reward, state, terminal = env:step(action)
    episodeScore = episodeScore + reward

    display:recordAndDisplay(agent, state, step)
    -- Increment evaluation step counter
    step = step + 1
  end
  log.info('Final Score: ' .. episodeScore)

  display:createVideo()

end
