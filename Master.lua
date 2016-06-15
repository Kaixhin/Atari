local classic = require 'classic'
local signal = require 'posix.signal'
local Singleton = require 'structures/Singleton'
local Agent = require 'Agent'
local Display = require 'Display'
local Validation = require 'Validation'

local Master = classic.class('Master')

-- Sets up environment and agent
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
  -- Extra safety check for Catch
  if not opt.ale then -- TODO: Remove eventually
    opt.height, opt.width = stateSpec[2][2], stateSpec[2][3]
  end
  -- Set up fake training mode (if needed)
  if not self.env.training then
    self.env.training = function() end
  end
  -- Set up fake evaluation mode (if needed)
  if not self.env.evaluate then
    self.env.evaluate = function() end
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

-- Trains agent
function Master:train()
  log.info('Training mode')

  -- Catch CTRL-C to save
  self:catchSigInt()

  local reward, state, terminal = 0, self.env:start(), false

  -- Set environment and agent to training mode
  self.env:training()
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
    if not self.opt.noValidation and step >= self.opt.learnStart and step % self.opt.valFreq == 0 then
      self.validation:validate() -- Sets env and agent to evaluation mode and then back to training mode

      log.info('Resuming training')
      -- Start new game (as previous one was interrupted)
      reward, state, terminal = 0, self.env:start(), false
      episodeScore = reward
    end
  end

  log.info('Finished training')
end

function Master:evaluate()
  self.validation:evaluate() -- Sets env and agent to evaluation mode
end

-- Sets up SIGINT (Ctrl+C) handler to save network before quitting
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
