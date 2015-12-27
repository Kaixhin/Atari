-- TODO: Confirm nomenclature for parameters - a frame is a step in ALE, a time step is consecutive frames treated atomically by the agent
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
cmd:option('-game', 'pong', 'Name of Atari ROM (stored in "roms" directory)')
-- Training vs. evaluate mode
cmd:option('-mode', 'train', '"train" or "eval" mode')
-- Screen preprocessing options
cmd:option('-height', 84, 'Resized screen height')
cmd:option('-width', 84, 'Resize screen width')
cmd:option('-colorSpace', 'y', 'Colour space conversion (screen is RGB): rgb|y|lab|yuv|hsl|hsv|nrgb')
--cmd:option('-agent_params', 'hist_len=4,update_freq=4,n_replay=1', 'string of agent parameters') -- TODO: Utilise
-- Experience replay options
cmd:option('-memSize', 1e6, 'Experience replay memory size (number of tuples)')
cmd:option('-memSampleFreq', 4, 'Memory sample frequency')
--cmd:option('-bufferSize', 512, 'Memory buffer size')
cmd:option('-memPriority', 'proportional', 'Type of prioritised experience replay: none|rank|proportional')
cmd:option('-alpha', 0.65, 'Prioritised experience replay exponent α') -- Best vals are rank = 0.7, proportional = 0.6
cmd:option('-betaZero', 0.45, 'Initial value of importance-sampling exponent β') -- Best vals are rank = 0.5, proportional = 0.4
-- Reinforcement learning parameters
cmd:option('-gamma', 0.99, 'Discount rate γ')
cmd:option('-eta', 7e-5, 'Learning rate η') -- Accounts for prioritied experience sampling but not duel
cmd:option('-epsilonStart', 1, 'Initial value of greediness ε')
cmd:option('-epsilonEnd', 0.01, 'Final value of greediness ε')
cmd:option('-epsilonSteps', 1e6, 'Number of steps to linearly decay epsilonStart to epsilonEnd') -- Usually same as memory size
cmd:option('-tau', 30000, 'Steps between target net updates τ') -- Larger for duel
cmd:option('-rewardClamp', 1, 'Clamps reward magnitude')
cmd:option('-tdClamp', 1, 'Clamps TD-error δ magnitude')
cmd:option('-PALpha', 0.9, 'Persistent advantage learning parameter α')
-- Training options
cmd:option('-optimiser', 'rmsprop', 'Training algorithm')
cmd:option('-momentum', 0.95, 'SGD momentum')
cmd:option('-batchSize', 32, 'Minibatch size')
cmd:option('-steps', 5e7, 'Training iterations (steps)') -- Equivalent to standard 200 million frames for DQN experiments
cmd:option('-learnStart', 50000, 'Number of steps after which learning starts')
-- Evaluation options
cmd:option('-valFreq', 250000, 'Validation frequency (by number of steps)')
cmd:option('-valSteps', 12500, 'Number of steps to use for validation') -- Usually 125000
--cmd:option('-valSize', 500, 'Number of transitions to use for validation')
-- alewrap options
cmd:option('-actrep', 4, 'Times to repeat action')
cmd:option('-random_starts', 30, 'Play no-op action between 1 and random_starts number of times at the start of each training episode')
-- Experiment options
cmd:option('-_id', 'ID', 'ID of experiment (used to store saved results)')
cmd:option('-network', '', 'Saved DQN file (DQN.t7)')
local opt = cmd:parse(arg)

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
-- Enable memory management
torch.setheaptracking(true)
-- Set number of BLAS threads
torch.setnumthreads(opt.threads)
-- Set default Tensor type (float is more efficient than double)
torch.setdefaulttensortype(opt.tensorType)
-- Set manual seeds using random numbers to reduce correlations
math.randomseed(opt.seed)
torch.manualSeed(math.random(1, 1e6))
-- GPU setup
if opt.gpu > 0 then
  log.info('Setting up GPU')
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
  cutorch.manualSeedAll(torch.random())
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
local DQN = agent.create(gameEnv, opt)
-- Load saved agent if specified
if paths.filep(opt.network) then
  log.info('Loading pretrained network')
  DQN:load(opt.network)
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
  opt.epsilon:mul(-1):add(opt.epsilonStart)
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

    -- Observe and choose next action (index)
    local actionIndex = DQN:observe(screen)
    if not terminal then
      -- Act on environment and learn
      screen, reward, terminal = DQN:act(actionIndex)
      cumulativeReward = cumulativeReward + reward
    else
      -- Print score for episode
      log.info('Episode ' .. episode .. ' | Score: ' .. cumulativeReward .. ' | Steps: ' .. step .. '/' .. opt.steps)

      -- Start a new episode
      episode = episode + 1
      if opt.random_starts > 0 then
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
      -- Track total score for validation
      local totalValScore = reward

      for valStep = 1, opt.valSteps do
        -- Observe and choose next action (index)
        local actionIndex = DQN:observe(screen)
        if not terminal then
          -- Act on environment
          screen, reward, terminal = DQN:act(actionIndex)
          -- Track scores
          cumulativeReward = cumulativeReward + reward
          totalValScore = totalValScore + reward
        else
          -- Print score for episode
          log.info('Val Episode ' .. valEpisode .. ' | Score: ' .. cumulativeReward .. ' | Steps: ' .. valStep .. '/' .. opt.valSteps)

          -- Start a new episode
          valEpisode = valEpisode + 1
          if opt.random_starts > 0 then
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

      -- Check total score against best
      log.info('Total Score: ' .. totalValScore)
      if totalValScore > bestValScore then
        log.info('New best score')
        bestValScore = totalValScore

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
    local actionIndex = DQN:observe(screen)
    -- Act on environment
    screen, reward, terminal = DQN:act(actionIndex)
    cumulativeReward = cumulativeReward + reward

    if qt then
      image.display({image=screen, win=window})
    end
  end
  log.info('Final score: ' .. cumulativeReward)
end
