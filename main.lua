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
-- Model options
cmd:option('-height', 84, 'Height to resize screen to')
cmd:option('-width', 84, 'Width to resize screen to')
--cmd:option('-agent_params', 'hist_len=4,update_freq=4,n_replay=1,ncols=1', 'string of agent parameters') -- TODO: Utilise
-- Experience replay options
cmd:option('-memSize', 1e6, 'Experience replay memory size (number of tuples)')
cmd:option('-memSampleFreq', 4, 'Memory sample frequency')
--cmd:option('-bufferSize', 512, 'Memory buffer size')
cmd:option('-alpha', 1, 'Prioritised experience replay exponent α')
cmd:option('-betaZero', 1, 'Initial value of importance-sampling exponent β')
-- Reinforcement learning parameters
cmd:option('-gamma', 0.99, 'Discount rate γ')
cmd:option('-eta', 7e-5, 'Learning rate η') -- Accounts for prioritied experience sampling but not duel
cmd:option('-epsilonStart', 1, 'Initial value of greediness ε')
cmd:option('-epsilonEnd', 0.01, 'Final value of greediness ε')
cmd:option('-epsilonSteps', 1e6, 'Number of steps to linearly decay epsilonStart to epsilonEnd')
cmd:option('-tau', 30000, 'Steps between target net updates τ') -- Larger for duel
cmd:option('-rewardClamp', 1, 'Clamps reward magnitude')
cmd:option('-tdClamp', 1, 'Clamps TD-error δ magnitude')
-- Training options
cmd:option('-optimiser', 'rmsprop', 'Training algorithm')
cmd:option('-momentum', 0.95, 'SGD momentum')
cmd:option('-batchSize', 32, 'Minibatch size')
cmd:option('-steps', 5e7, 'Training iterations (steps)')
--cmd:option('-learnStart', 50000, 'Number of steps after which learning starts')
-- Evaluation options
cmd:option('-evalFreq', 1e6, 'Evaluation frequency (by number of steps)')
cmd:option('-evalSize', 500, 'Number of transitions to use for validation')
-- alewrap options
cmd:option('-actrep', 4, 'Times to repeat action')
cmd:option('-random_starts', 30, 'Play no-op action between 1 and random_starts number of times at the start of each training episode')
local opt = cmd:parse(arg)

-- Torch setup
-- Enable memory management
torch.setheaptracking(true)
-- Set number of BLAS threads
torch.setnumthreads(opt.threads)
-- Set default Tensor type (float is more efficient than double)
torch.setdefaulttensortype(opt.tensorType)
-- Set manual seeds using random numbers to reduce correlations
math.randomseed(opt.seed)
torch.manualSeed(math.random(1, 10000))
-- GPU setup
if opt.gpu > 0 then
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
  cutorch.manualSeedAll(torch.random())
end

-- Initialise Arcade Learning Environment
local gameEnv = environment.init(opt)

-- Create DQN agent
local DQN = agent.create(gameEnv, opt)

-- Start gaming
local screen, reward, terminal = gameEnv:newGame()
-- Activate display if using QT
local window
if qt then
  window = image.display({image=screen})
end

if opt.mode == 'train' then
  -- Create ε decay vector
  opt.epsilon = torch.linspace(opt.epsilonEnd, opt.epsilonStart, opt.epsilonSteps)
  opt.epsilon:mul(-1):add(opt.epsilonStart)
  local epsilonFinal = torch.Tensor(opt.steps - opt.epsilonSteps):fill(opt.epsilonEnd)
  opt.epsilon = torch.cat(opt.epsilon, epsilonFinal)
  -- Create β growth vector
  opt.beta = torch.linspace(opt.betaZero, 1, opt.steps)

  -- Set agent (and hence environment steps) to training mode
  DQN:training()

  -- Training loop
  for step = 1, opt.steps do
    opt.step = step -- Pass step number to agent for use in training

    -- Observe and choose next action (index)
    local actionIndex = DQN:observe(screen)
    if not terminal then
      -- Act on environment and learn
      screen, reward, terminal = DQN:act(actionIndex)
    else
      -- Start a new episode
      if opt.random_starts > 0 then
        screen, reward, terminal = gameEnv:nextRandomGame()
      else
        screen, reward, terminal = gameEnv:newGame()
      end
    end

    -- Update display
    if qt then
      image.display({image=screen, win=window})
    end

    if step % opt.evalFreq then
      DQN:evaluate()
      -- TODO: Perform evaluation
      -- TODO: Save best parameters
      DQN:training()
    end
  end
elseif opt.mode == 'eval' then
  -- Set agent (and hence environment steps) to evaluation mode
  DQN:evaluate()

  -- Play one game (episode)
  while not terminal do
    local actionIndex = DQN:observe(screen)
    -- Act on environment
    screen, reward, terminal = DQN:act(actionIndex)

    if qt then
      image.display({image=screen, win=window})
    end
  end
end
