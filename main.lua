local signal = require 'posix.signal'
local _ = require 'moses'
require 'logroll'
local image = require 'image'
local environment = require 'environment'
local agent = require 'agent'
local evaluator = require 'evaluator'

-- Detect QT for image display
local qt = pcall(require, 'qt')

local cmd = torch.CmdLine()
-- Base Torch7 options
cmd:option('-seed', 123, 'Random seed')
cmd:option('-threads', 4, 'Number of BLAS threads')
cmd:option('-tensorType', 'torch.FloatTensor', 'Default tensor type')
cmd:option('-gpu', 1, 'GPU device ID (0 to disable)')
-- Game
cmd:option('-game', 'space_invaders', 'Name of Atari ROM (stored in "roms" directory)')
-- Training vs. evaluate mode
cmd:option('-mode', 'train', 'Train vs. test mode: train|eval')
-- Screen preprocessing options
cmd:option('-height', 84, 'Resized screen height')
cmd:option('-width', 84, 'Resize screen width')
cmd:option('-colorSpace', 'y', 'Colour space conversion (screen is RGB): rgb|y|lab|yuv|hsl|hsv|nrgb')
-- Agent options
cmd:option('-histLen', 4, 'Number of consecutive states processed')
-- Experience replay options
cmd:option('-memSize', 1e6, 'Experience replay memory size (number of tuples)')
cmd:option('-memSampleFreq', 4, 'Memory sample frequency')
cmd:option('-memNReplay', 1, 'Number of times to replay per learning step')
cmd:option('-memPriority', 'none', 'Type of prioritised experience replay: none|rank|proportional')
cmd:option('-alpha', 0.65, 'Prioritised experience replay exponent α') -- Best vals are rank = 0.7, proportional = 0.6
cmd:option('-betaZero', 0.45, 'Initial value of importance-sampling exponent β') -- Best vals are rank = 0.5, proportional = 0.4
-- Reinforcement learning parameters
cmd:option('-gamma', 0.99, 'Discount rate γ')
cmd:option('-eta', 7e-5, 'Learning rate η') -- Accounts for prioritied experience sampling but not duel
cmd:option('-epsilonStart', 1, 'Initial value of greediness ε')
cmd:option('-epsilonEnd', 0.01, 'Final value of greediness ε')
cmd:option('-epsilonSteps', 1e6, 'Number of steps to linearly decay epsilonStart to epsilonEnd') -- Usually same as memory size
cmd:option('-tau', 30000, 'Steps between target net updates τ') -- Larger for duel than for standard DQN
cmd:option('-rewardClip', 1, 'Clips reward magnitude at rewardClip')
cmd:option('-tdClip', 1, 'Clips TD-error δ magnitude at tdClip')
cmd:option('-doubleQ', 'true', 'Use Double-Q learning')
-- Note from Georg Ostrovski: The advantage operators and Double DQN are not entirely orthogonal as the increased action gap seems to reduce the statistical bias that leads to value over-estimation in a similar way that Double DQN does
cmd:option('-PALpha', 0.9, 'Persistent advantage learning parameter α (0 to disable)')
-- Training options
cmd:option('-optimiser', 'rmsprop', 'Training algorithm')
cmd:option('-momentum', 0.95, 'SGD momentum')
cmd:option('-batchSize', 32, 'Minibatch size')
cmd:option('-steps', 5e7, 'Training iterations (steps)') -- Frame := step in ALE; Time step := consecutive frames treated atomically by the agent
cmd:option('-learnStart', 50000, 'Number of steps after which learning starts')
-- Evaluation options
cmd:option('-valFreq', 250000, 'Validation frequency (by number of steps)')
cmd:option('-valSteps', 125000, 'Number of steps to use for validation')
--cmd:option('-valSize', 500, 'Number of transitions to use for validation')
-- ALEWrap options
cmd:option('-actRep', 4, 'Times to repeat action') -- Independent of history length
cmd:option('-randomStarts', 30, 'Play no-op action between 1 and randomStarts number of times at the start of each training episode')
cmd:option('-poolFrmsType', 'max', 'Type of pooling over frames: max|mean')
cmd:option('-poolFrmsSize', 2, 'Size of pooling over frames')
-- Experiment options
cmd:option('-_id', '', 'ID of experiment (used to store saved results, defaults to game name)')
cmd:option('-network', '', 'Saved DQN file to load (DQN.t7)')
local opt = cmd:parse(arg)
-- Process boolean options (Torch fails to accept false on the command line)
opt.doubleQ = opt.doubleQ and true or false

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

-- Torch setup
log.info('Setting up Torch7')
-- Set number of BLAS threads
torch.setnumthreads(opt.threads)
-- Set default Tensor type (float is more efficient than double)
torch.setdefaulttensortype(opt.tensorType)
-- Set manual seeds using random numbers to reduce correlations
math.randomseed(opt.seed)
torch.manualSeed(math.random(1, 1e6))

-- Tensor creation function for removing need to cast to CUDA if GPU is enabled
opt.Tensor = function(...)
  return torch.Tensor(...)
end

-- GPU setup
if opt.gpu > 0 then
  log.info('Setting up GPU')
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
  cutorch.manualSeedAll(torch.random())
  -- Replace tensor creation function
  opt.Tensor = function(...)
    return torch.CudaTensor(...)
  end
end

-- Work out number of colour channels
if not _.contains({'rgb', 'y', 'lab', 'yuv', 'hsl', 'hsv', 'nrgb'}, opt.colorSpace) then
  log.error('Unsupported colour space for conversion')
  error('Unsupported colour space for conversion')
end
opt.nChannels = opt.colorSpace == 'y' and 1 or 3

-- Initialise Arcade Learning Environment
log.info('Setting up ALE')
local gameEnv = environment.init(opt)

-- Create DQN agent
log.info('Creating DQN')
local DQN = agent.create(gameEnv, opt)
if paths.filep(opt.network) then
  -- Load saved agent if specified
  log.info('Loading pretrained network')
  DQN:load(opt.network)
elseif paths.filep(paths.concat('experiments', opt._id, 'DQN.t7')) then
  -- Ask to load saved agent if found in experiment folder
  log.info('Saved network found - load (y/n)?')
  if io.read() == 'y' then
    log.info('Loading pretrained network')
    DQN:load(paths.concat('experiments', opt._id, 'DQN.t7'))
  end
end

-- Start gaming
log.info('Starting game: ' .. opt.game)
local screen, reward, terminal = gameEnv:newGame()
local cumulativeReward = reward
local bestValScore = -math.huge -- Best validation score
-- Activate display if using QT
local window = qt and image.display({image=screen})

if opt.mode == 'train' then
  log.info('Training mode')

  -- Set up SIGINT (Ctrl+C) handler to save network before quitting
  signal.signal(signal.SIGINT, function(signum)
    log.warn('SIGINT received')
    log.info('Save network (y/n)?')
    if io.read() == 'y' then
      log.info('Saving network')
      DQN:save(paths.concat('experiments', opt._id))
    end
    log.warn('Exiting')
    os.exit(128 + signum)
  end)

  -- Check prioritised experience replay options
  if opt.memPriority == 'rank' then
    log.info('Rank-based prioritised experience replay is not implemented, switching to proportional')
    opt.memPriority = 'proportional'
  elseif opt.memPriority ~= 'none' and opt.memPriority ~= 'proportional' then
    log.error('Unrecognised type of prioritised experience replay')
    error('Unrecognised type of prioritised experience replay')
  end

  -- Create ε decay vector
  opt.epsilon = torch.linspace(opt.epsilonEnd, opt.epsilonStart, opt.epsilonSteps)
  opt.epsilon:neg():add(opt.epsilonStart)
  local epsilonFinal = torch.Tensor(opt.steps - opt.epsilonSteps):fill(opt.epsilonEnd)
  opt.epsilon = torch.cat(opt.epsilon, epsilonFinal)
  -- Create β growth vector
  opt.beta = torch.linspace(opt.betaZero, 1, opt.steps)

  -- Set agent (and hence environment steps) to training mode
  DQN:training()
  -- Keep track of episodes
  local episode = 1

  -- Training loop
  for step = 1, opt.steps do
    opt.step = step -- Pass step number to agent for use in training

    -- Observe results of previous transition (r, s', terminal') and choose next action (index)
    local actionIndex = DQN:observe(reward, screen, terminal) -- As results received, learn in training mode
    if not terminal then
      -- Act on environment (to cause transition)
      screen, reward, terminal = DQN:act(actionIndex)
      cumulativeReward = cumulativeReward + reward
    else
      -- Print score for episode
      log.info('Episode ' .. episode .. ' | Score: ' .. cumulativeReward .. ' | Steps: ' .. step .. '/' .. opt.steps)

      -- Start a new episode
      episode = episode + 1
      if opt.randomStarts > 0 then
        screen, reward, terminal = gameEnv:nextRandomGame()
      else
        screen, reward, terminal = gameEnv:newGame()
      end
      cumulativeReward = reward -- Refresh cumulative reward
    end

    -- Update display
    if qt then
      image.display({image=screen, win=window})
    end

    -- Trigger learning after a while (wait to accumulate experience)
    if step == opt.learnStart then
      log.info('Learning started')
    end

    if step % opt.valFreq == 0 and step >= opt.learnStart then
      -- TODO: Include TD-error δ squared loss as metric
      log.info('Validating')
      DQN:evaluate()

      -- Start new game
      screen, reward, terminal = gameEnv:newGame()
      local valEpisode = 1
      -- Reset cumulative reward
      cumulativeReward = reward
      -- Track total/average score for validation
      local valScore = reward

      for valStep = 1, opt.valSteps do
        -- Observe and choose next action (index)
        local actionIndex = DQN:observe(reward, screen, terminal)
        if not terminal then
          -- Act on environment
          screen, reward, terminal = DQN:act(actionIndex)
          -- Track scores
          cumulativeReward = cumulativeReward + reward
          valScore = valScore + reward
        else
          -- Print score for episode
          log.info('Val Episode ' .. valEpisode .. ' | Score: ' .. cumulativeReward .. ' | Steps: ' .. valStep .. '/' .. opt.valSteps)

          -- Start a new episode
          valEpisode = valEpisode + 1
          if opt.randomStarts > 0 then
            screen, reward, terminal = gameEnv:nextRandomGame()
          else
            screen, reward, terminal = gameEnv:newGame()
          end
          cumulativeReward = reward -- Refresh cumulative reward
        end

        -- Update display
        if qt then
          image.display({image=screen, win=window})
        end
      end

      -- Check average score against best
      valScore = valScore/valEpisode
      log.info('Average Score: ' .. valScore)
      if valScore > bestValScore then
        log.info('New best score')
        bestValScore = valScore

        log.info('Saving network')
        DQN:save(paths.concat('experiments', opt._id))
      end

      log.info('Resuming training')
      DQN:training()
      -- Start new game (as previous episode was interrupted)
      screen, reward, terminal = gameEnv:newGame()
      cumulativeReward = reward
    end
  end
elseif opt.mode == 'eval' then
  log.info('Evaluation mode')
  -- Set agent (and hence environment steps) to evaluation mode
  DQN:evaluate()

  -- Play one game (episode)
  while not terminal do
    -- Observe and choose next action (index)
    local actionIndex = DQN:observe(reward, screen, terminal)
    -- Act on environment
    screen, reward, terminal = DQN:act(actionIndex)
    cumulativeReward = cumulativeReward + reward

    if qt then
      image.display({image=screen, win=window})
    end
  end
  log.info('Final score: ' .. cumulativeReward)
end
