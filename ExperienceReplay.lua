local Display = require 'Display'
local signal = require 'posix.signal'
local Singleton = require 'structures/Singleton'
local Agent = require 'Agent'
local classic = require 'classic'
local Validation = require 'Validation'

local ExperienceReplay = classic.class('ExperienceReplay')

function ExperienceReplay:_init(opt)
  self.opt = opt

  -- Set up singleton global object for transferring step
  self.globals = Singleton({step = 1}) -- Initial step

  ----- Environment + Agent Setup -----

  -- Initialise Catch or Arcade Learning Environment
  log.info('Setting up ' .. (opt.ale and 'Arcade Learning Environment' or 'Catch'))
  if opt.ale then
    local Atari = require 'rlenvs.Atari'
    self.env = Atari(opt)
    local stateSpec = self.env:getStateSpec()

    -- Provide original channels, height and width for resizing from
    opt.origChannels, opt.origHeight, opt.origWidth = table.unpack(stateSpec[2])
  else
    local Catch = require 'rlenvs.Catch'
    self.env = Catch()
    local stateSpec = self.env:getStateSpec()
    
    -- Provide original channels, height and width for resizing from
    opt.origChannels, opt.origHeight, opt.origWidth = table.unpack(stateSpec[2])

    -- Adjust height and width
    opt.height, opt.width = stateSpec[2][2], stateSpec[2][3]
  end

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

  -- Start gaming
  log.info('Starting game: ' .. opt.game)
  local state = self.env:start()
  self.display = Display(opt, state)

  self.validation = Validation(opt, self.agent, self.env, self.display)

  classic.strict(self)
end


function ExperienceReplay:train()
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
      self.validation:validate()

      log.info('Resuming training')
      -- Set environment and agent to training mode
      if self.opt.ale then self.env:training() end
      self.agent:training()

      -- Start new game (as previous one was interrupted)
      reward, state, terminal = 0, self.env:start(), false
      episodeScore = reward
    end
  end

  log.info('Finished training')
end


function ExperienceReplay:evaluate()
  self.validation:evaluate()
end


-- Set up SIGINT (Ctrl+C) handler to save network before quitting
function ExperienceReplay:catchSigInt()
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


return ExperienceReplay
