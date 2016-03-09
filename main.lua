----- General Setup -----

local signal = require 'posix.signal'
local _ = require 'moses'
local image = require 'image'
local gnuplot = require 'gnuplot'
local Singleton = require 'structures/Singleton'
local Agent = require 'Agent'
local evaluator = require 'evaluator'
require 'logroll'

-- Detect QT for image display
local qt = pcall(require, 'qt')
-- Detect and use GPU 1 by default
local cuda = pcall(require, 'cutorch')
-- Create log10 for Lua 5.2
if not math.log10 then
  math.log10 = function(x)
    return math.log(x, 10)
  end
end

local cmd = torch.CmdLine()
-- Base Torch7 options
cmd:option('-seed', 1, 'Random seed')
cmd:option('-threads', 4, 'Number of BLAS threads')
cmd:option('-tensorType', 'torch.FloatTensor', 'Default tensor type')
cmd:option('-gpu', cuda and 1 or 0, 'GPU device ID (0 to disable)')
-- Game
cmd:option('-game', 'catch', 'Name of Atari ROM (stored in "roms" directory)') -- Uses "Catch" env by default
-- Training vs. evaluate mode
cmd:option('-mode', 'train', 'Train vs. test mode: train|eval')
-- Screen preprocessing options
cmd:option('-height', 84, 'Resized screen height')
cmd:option('-width', 84, 'Resize screen width')
cmd:option('-colorSpace', 'y', 'Colour space conversion (screen is RGB): rgb|y|lab|yuv|hsl|hsv|nrgb')
-- Agent options
cmd:option('-histLen', 4, 'Number of consecutive states processed')
cmd:option('-duel', 'true', 'Use dueling network architecture (learns advantage function)')
cmd:option('-bootstraps', 10, 'Number of bootstrap heads (0 to disable)')
--cmd:option('-bootstrapMask', 1, 'Independent probability of masking a transition for each bootstrap head ~ Ber(bootstrapMask) (1 to disable)')
-- Experience replay options
cmd:option('-memSize', 1e6, 'Experience replay memory size (number of tuples)')
cmd:option('-memSampleFreq', 4, 'Interval of steps between sampling from memory to learn')
cmd:option('-memNSamples', 1, 'Number of times to sample per learning step')
cmd:option('-memPriority', 'rank', 'Type of prioritised experience replay: none|rank|proportional')
cmd:option('-alpha', 0.65, 'Prioritised experience replay exponent α') -- Best vals are rank = 0.7, proportional = 0.6
cmd:option('-betaZero', 0.45, 'Initial value of importance-sampling exponent β') -- Best vals are rank = 0.5, proportional = 0.4
-- Reinforcement learning parameters
cmd:option('-gamma', 0.99, 'Discount rate γ')
cmd:option('-epsilonStart', 1, 'Initial value of greediness ε')
cmd:option('-epsilonEnd', 0.01, 'Final value of greediness ε') -- Tuned DDQN final greediness (1/10 that of DQN)
cmd:option('-epsilonSteps', 1e6, 'Number of steps to linearly decay epsilonStart to epsilonEnd') -- Usually same as memory size
cmd:option('-tau', 30000, 'Steps between target net updates τ') -- Tuned DDQN target net update interval (3x that of DQN)
cmd:option('-rewardClip', 1, 'Clips reward magnitude at rewardClip (0 to disable)')
cmd:option('-tdClip', 1, 'Clips TD-error δ magnitude at tdClip (0 to disable)')
cmd:option('-doubleQ', 'true', 'Use Double Q-learning')
-- Note from Georg Ostrovski: The advantage operators and Double DQN are not entirely orthogonal as the increased action gap seems to reduce the statistical bias that leads to value over-estimation in a similar way that Double DQN does
cmd:option('-PALpha', 0.9, 'Persistent advantage learning parameter α (0 to disable)')
-- Training options
cmd:option('-optimiser', 'rmspropm', 'Training algorithm') -- RMSProp with momentum as found in "Generating Sequences With Recurrent Neural Networks"
cmd:option('-eta', 0.00025/4, 'Learning rate η') -- Prioritied experience replay learning rate (1/4 that of DQN; does not account for Duel as well)
cmd:option('-momentum', 0.95, 'Gradient descent momentum')
cmd:option('-batchSize', 32, 'Minibatch size')
cmd:option('-steps', 5e7, 'Training iterations (steps)') -- Frame := step in ALE; Time step := consecutive frames treated atomically by the agent
cmd:option('-learnStart', 50000, 'Number of steps after which learning starts')
cmd:option('-gradClip', 10, 'Clips L2 norm of gradients at gradClip (0 to disable)')
-- Evaluation options
cmd:option('-progFreq', 10000, 'Interval of steps between reporting progress')
cmd:option('-valFreq', 250000, 'Interval of steps between validating agent') -- valFreq steps is used as an epoch, hence #epochs = steps/valFreq
cmd:option('-valSteps', 125000, 'Number of steps to use for validation')
cmd:option('-valSize', 500, 'Number of transitions to use for calculating validation statistics')
-- ALEWrap options
cmd:option('-actRep', 4, 'Times to repeat action') -- Independent of history length
cmd:option('-randomStarts', 30, 'Max number of no-op actions played before presenting the start of each training episode')
cmd:option('-poolFrmsType', 'max', 'Type of pooling over previous emulator frames: max|mean')
cmd:option('-poolFrmsSize', 2, 'Number of emulator frames to pool over')
-- Experiment options
cmd:option('-_id', '', 'ID of experiment (used to store saved results, defaults to game name)')
cmd:option('-network', '', 'Saved network weights file to load (weights.t7)')
cmd:option('-verbose', 'false', 'Log info for every episode (only in train mode)')
cmd:option('-saliency', 'none', 'Display saliency maps (requires QT): none|normal|guided|deconvnet')
cmd:option('-record', 'false', 'Record screen (only in eval mode)')
local opt = cmd:parse(arg)

-- Process boolean options (Torch fails to accept false on the command line)
opt.duel = opt.duel == 'true' or false
opt.doubleQ = opt.doubleQ == 'true' or false
opt.verbose = opt.verbose == 'true' or false
opt.record = opt.record == 'true' or false

-- Set ID as game name if not set
if opt._id == '' then
  opt._id = opt.game
end

-- Create experiment directory
if not paths.dirp('experiments') then
  paths.mkdir('experiments')
end
paths.mkdir(paths.concat('experiments', opt._id))

-- Set up logs
local flog = logroll.file_logger(paths.concat('experiments', opt._id, 'log.txt'))
local plog = logroll.print_logger()
log = logroll.combine(flog, plog)

-- Calculate number of colour channels
if not _.contains({'rgb', 'y', 'lab', 'yuv', 'hsl', 'hsv', 'nrgb'}, opt.colorSpace) then
  log.error('Unsupported colour space for conversion')
  error('Unsupported colour space for conversion')
end
opt.nChannels = opt.colorSpace == 'y' and 1 or 3

-- Check start of learning occurs after at least one minibatch of data has been collected
if opt.learnStart <= opt.batchSize then
  log.error('learnStart must be greater than batchSize')
  error('learnStart must be greater than batchSize')
end

-- Check enough validation transitions will be collected before first validation
if opt.valFreq <= opt.valSize then
  log.error('valFreq must be greater than valSize')
  error('valFreq must be greater than valSize')
end

-- Check prioritised experience replay options
if not _.contains({'none', 'rank', 'proportional'}, opt.memPriority) then
  log.error('Unrecognised type of prioritised experience replay')
  error('Unrecognised type of prioritised experience replay')
end

-- Check start of learning occurs after at least 1/100 of memory has been filled
if opt.learnStart <= opt.memSize/100 then
  log.error('learnStart must be greater than memSize/100')
  error('learnStart must be greater than memSize/100')
end

-- Check memory size is multiple of 100 (makes prioritised sampling partitioning simpler)
if opt.memSize % 100 ~= 0 then
  log.error('memSize must be a multiple of 100')
  error('memSize must be a multiple of 100')
end

-- Check learning occurs after first progress report
if opt.learnStart < opt.progFreq then
  log.error('learnStart must be greater than progFreq')
  error('learnStart must be greater than progFreq')
end

-- Check saliency map options
if not _.contains({'none', 'normal', 'guided', 'deconvnet'}, opt.saliency) then
  log.error('Unrecognised method for visualising saliency maps')
  error('Unrecognised method for visualising saliency maps')
end


-- Torch setup
log.info('Setting up Torch7')
-- Use enhanced garbage collector
torch.setheaptracking(true)
-- Set number of BLAS threads
torch.setnumthreads(opt.threads)
-- Set default Tensor type (float is more efficient than double)
torch.setdefaulttensortype(opt.tensorType)
-- Set manual seed
torch.manualSeed(opt.seed)

-- Tensor creation function for removing need to cast to CUDA if GPU is enabled
opt.Tensor = function(...)
  return torch.Tensor(...)
end

-- GPU setup
if opt.gpu > 0 then
  log.info('Setting up GPU')
  cutorch.setDevice(opt.gpu)
  -- Set manual seeds using random numbers to reduce correlations
  cutorch.manualSeed(torch.random())
  -- Replace tensor creation function
  opt.Tensor = function(...)
    return torch.CudaTensor(...)
  end
end

-- Set up singleton global object for transferring step
local globals = Singleton({step = 1}) -- Initial step

-- Computes saliency map for display
local createSaliencyMap = function(state, agent)
  local screen
  
  -- Convert Catch screen to RGB
  if opt.game == 'catch' then
    screen = torch.repeatTensor(state, 3, 1, 1)
  else
    screen = state:select(1, 1):clone()
  end

  -- Use red channel for saliency map
  screen:select(1, 1):copy(agent.saliencyMap)

  return screen
end

----- Environment + Agent Setup -----

-- Initialise Catch or Arcade Learning Environment
opt.ale = opt.game ~= 'catch'
log.info('Setting up ' .. (opt.ale and 'Arcade Learning Environment' or 'Catch'))
local env, stateSpec
if opt.ale then
  local Atari = require 'rlenvs.Atari'
  env = Atari(opt)
  stateSpec = env:getStateSpec()

  -- Provide original channels, height and width for resizing from
  opt.origChannels, opt.origHeight, opt.origWidth = unpack(stateSpec[2])
else
  local Catch = require 'rlenvs.Catch'
  env = Catch()
  stateSpec = env:getStateSpec()
  
  -- Provide original channels, height and width for resizing from
  opt.origChannels, opt.origHeight, opt.origWidth = unpack(stateSpec[2])

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
elseif paths.filep(paths.concat('experiments', opt._id, 'agent.t7')) then
  -- Ask to load saved agent if found in experiment folder (resuming training)
  log.info('Saved agent found - load (y/n)?')
  if io.read() == 'y' then
    log.info('Loading saved agent')
    agent = torch.load(paths.concat('experiments', opt._id, 'agent.t7'))

    -- Reset globals (step) from agent
    Singleton.setInstance(agent.globals)
    globals = Singleton.getInstance()

    -- Switch saliency style
    agent:setSaliency(opt.saliency)
  end
end

----- Training / Evaluation -----

-- Start gaming
log.info('Starting game: ' .. opt.game)
local reward, state, terminal = 0, env:start(), false
local action

-- Activate display if using QT
local zoom = opt.ale and 1 or 4
local screen = state -- Use separate screen for displaying saliency maps
local window = qt and image.display({image=screen, zoom=zoom})


if opt.mode == 'train' then

  log.info('Training mode')

  -- Set up SIGINT (Ctrl+C) handler to save network before quitting
  signal.signal(signal.SIGINT, function(signum)
    log.warn('SIGINT received')
    log.info('Save agent (y/n)?')
    if io.read() == 'y' then
      log.info('Saving agent')
      torch.save(paths.concat('experiments', opt._id, 'agent.t7'), agent) -- Save agent to resume training
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
  local valEpisode, valEpisodeScore, valTotalScore
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

    -- Update display
    if qt then
      screen = opt.saliency ~= 'none' and createSaliencyMap(state, agent) or state
      image.display({image=screen, zoom=zoom, win=window})
    end

    -- Trigger learning after a while (wait to accumulate experience)
    if step == opt.learnStart then
      log.info('Learning started')
    end

    -- Report progress
    if step % opt.progFreq == 0 then
      log.info('Steps: ' .. string.format(stepStrFormat, step) .. '/' .. opt.steps)
      -- TODO: Report absolute weight and weight gradient values per module in policy network
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

        -- Update display
        if qt then
          screen = opt.saliency ~= 'none' and createSaliencyMap(state, agent) or state
          image.display({image=screen, zoom=zoom, win=window})
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
      -- Pass to agent (for storage and plotting)
      agent.valScores[#agent.valScores + 1] = valTotalScore

      -- Visualise convolutional filters
      agent:visualiseFilters()

      -- Use transitions sampled for validation to test performance
      local avgV, avgTdErr = agent:report()
      log.info('Average V: ' .. avgV)
      log.info('Average δ: ' .. avgTdErr)

      -- Save if best score achieved
      if valTotalScore > bestValScore then
        log.info('New best average score')
        bestValScore = valTotalScore

        log.info('Saving weights')
        agent:saveWeights(paths.concat('experiments', opt._id, 'weights.t7'))
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

  -- Set up recording
  if opt.record then
    -- Recreate scratch directory
    paths.rmall('scratch', 'yes')
    paths.mkdir('scratch')

    log.info('Recording screen')
  end

  -- Play one game (episode)
  local step = 1
  while not terminal do
    -- Observe and choose next action (index)
    action = agent:observe(reward, state, terminal)
    -- Act on environment
    reward, state, terminal = env:step(action)
    episodeScore = episodeScore + reward

    if qt or opt.record then
      -- Extract screen in RGB format for saving images for FFmpeg
      screen = opt.saliency ~= 'none' and createSaliencyMap(state, agent) or (opt.game == 'catch' and torch.repeatTensor(state, 3, 1, 1) or state:select(1, 1))
      if qt then
        image.display({image=screen, zoom=zoom, win=window})
      end
      if opt.record then
        image.save(paths.concat('scratch', opt.game .. '_' .. string.format('%06d', step) .. '.jpg'), screen)
      end
    end

    -- Increment evaluation step counter
    step = step + 1
  end
  log.info('Final Score: ' .. episodeScore)

  -- Export recording as video
  if opt.record then
    log.info('Recorded screen')

    -- Create videos directory
    if not paths.dirp('videos') then
      paths.mkdir('videos')
    end

    -- Use FFmpeg to create a video from the screens
    log.info('Creating video')
    local fps = opt.game == 'catch' and 10 or 60
    os.execute('ffmpeg -framerate ' .. fps .. ' -start_number 1 -i scratch/' .. opt.game .. '_%06d.jpg -c:v libvpx-vp9 -crf 0 -b:v 0 videos/' .. opt.game .. '.webm')
    log.info('Created video')

    -- Clear scratch space
    paths.rmall('scratch', 'yes')
  end

end
