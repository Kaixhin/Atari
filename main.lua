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
-- Experience replay options
cmd:option('-memSize', 1e6, 'Experience replay memory size (number of tuples)')
cmd:option('-memSampleFreq', 4, 'Interval of steps between sampling from memory to learn')
cmd:option('-memNSamples', 1, 'Number of times to sample per learning step')
cmd:option('-memPriority', 'none', 'Type of prioritised experience replay: none|rank|proportional')
cmd:option('-alpha', 0.65, 'Prioritised experience replay exponent α') -- Best vals are rank = 0.7, proportional = 0.6
cmd:option('-betaZero', 0.45, 'Initial value of importance-sampling exponent β') -- Best vals are rank = 0.5, proportional = 0.4
-- Reinforcement learning parameters
cmd:option('-gamma', 0.99, 'Discount rate γ')
cmd:option('-epsilonStart', 1, 'Initial value of greediness ε')
cmd:option('-epsilonEnd', 0.01, 'Final value of greediness ε') -- Tuned for DDQN (1/5 that of DQN)
cmd:option('-epsilonSteps', 1e6, 'Number of steps to linearly decay epsilonStart to epsilonEnd') -- Usually same as memory size
cmd:option('-tau', 30000, 'Steps between target net updates τ') -- Tuned for duel (3x that of DQN)
cmd:option('-rewardClip', 1, 'Clips reward magnitude at rewardClip (0 to disable)')
cmd:option('-tdClip', 1, 'Clips TD-error δ magnitude at tdClip (0 to disable)')
cmd:option('-doubleQ', 'true', 'Use Double Q-learning')
-- Note from Georg Ostrovski: The advantage operators and Double DQN are not entirely orthogonal as the increased action gap seems to reduce the statistical bias that leads to value over-estimation in a similar way that Double DQN does
cmd:option('-PALpha', 0, 'Persistent advantage learning parameter α (0 to disable)') -- TODO: Reset to 0.9 eventually (reasonably incompatible with Duel/PER)
-- Training options
cmd:option('-optimiser', 'rmspropm', 'Training algorithm') -- RMSProp with momentum as found in "Generating Sequences With Recurrent Neural Networks"
cmd:option('-eta', 0.002, 'Learning rate η') -- Prioritied experience replay learning rate (does not account for duel as well) x batch size (this code divides grads by batch size)
cmd:option('-momentum', 0.95, 'Gradient descent momentum')
cmd:option('-batchSize', 32, 'Minibatch size')
cmd:option('-steps', 5e7, 'Training iterations (steps)') -- Frame := step in ALE; Time step := consecutive frames treated atomically by the agent
cmd:option('-learnStart', 50000, 'Number of steps after which learning starts')
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
cmd:option('-verbose', 'false', 'Log info for every training episode')
cmd:option('-saliency', 'none', 'Display saliency maps (requires QT): none|normal|guided|deconvnet')
local opt = cmd:parse(arg)

-- Process boolean options (Torch fails to accept false on the command line)
opt.duel = opt.duel == 'true' or false
opt.doubleQ = opt.doubleQ == 'true' or false
opt.verbose = opt.verbose == 'true' or false

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

-- Check prioritised experience replay options
if not _.contains({'none', 'rank', 'proportional'}, opt.memPriority) then
  log.error('Unrecognised type of prioritised experience replay')
  error('Unrecognised type of prioritised experience replay')
end
opt.memPriority = 'none' -- TODO: Remove once other types are supported

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
-- Set manual seeds using random numbers to reduce correlations
math.randomseed(opt.seed)
torch.manualSeed(math.random(1, 1e3))

-- Tensor creation function for removing need to cast to CUDA if GPU is enabled
opt.Tensor = function(...)
  return torch.Tensor(...)
end

-- GPU setup
if opt.gpu > 0 then
  log.info('Setting up GPU')
  cutorch.setDevice(opt.gpu)
  cutorch.manualSeedAll(torch.random())
  -- Replace tensor creation function
  opt.Tensor = function(...)
    return torch.CudaTensor(...)
  end
end

-- Set up singleton global object for transferring step
local globals = Singleton({step = 1}) -- Initial step

-- Computes saliency map for display
local createSaliencyMap = function(state, agent)
  local screen = state
  
  -- Convert Catch screen to RGB
  if opt.game == 'catch' then
    screen = torch.repeatTensor(state, 3, 1, 1)
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

  -- TODO: Adjust parameters to better suit Catch
  opt.memSize = 1e4
  opt.optimiser = 'adam'
  opt.eta = 0.01
  opt.epsilonEnd = 0.05
  opt.epsilonSteps = 1e4
  opt.tau = 40
  opt.steps = 1e5
  opt.learnStart = 500
  opt.progFreq = 1000
  opt.valFreq = 12500
  opt.valSteps = 2400
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

  -- Training loop
  local initStep = globals.step -- Extract step
  for step = initStep, opt.steps do
    globals.step = step -- Pass step number to globals for use in training
    
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
        log.info('Steps: ' .. step .. '/' .. opt.steps .. ' | Episode ' .. episode .. ' | Score: ' .. episodeScore)
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
      log.info('Steps: ' .. step .. '/' .. opt.steps)
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
            log.info('[VAL] Steps: ' .. valStep .. '/' .. opt.valSteps .. ' | Episode ' .. valEpisode .. ' | Score: ' .. valEpisodeScore)
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

  -- Play one game (episode)
  while not terminal do
    -- Observe and choose next action (index)
    action = agent:observe(reward, state, terminal)
    -- Act on environment
    reward, state, terminal = env:step(action)
    episodeScore = episodeScore + reward

    if qt then
      screen = opt.saliency ~= 'none' and createSaliencyMap(state, agent) or state
      image.display({image=screen, zoom=zoom, win=window})
    end
  end
  log.info('Final Score: ' .. episodeScore)

end
