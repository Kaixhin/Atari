-- Parameters taken from (tuned) Double DQN paper: http://arxiv.org/pdf/1509.06461.pdf
local image = require 'image'
local environment = require 'environment'
local agent = require 'agent'

-- Detect QT for image display
local qt = pcall(require, 'qt')

local cmd = torch.CmdLine()
-- Base options
cmd:option('-seed', 123, 'Random seed')
cmd:option('-threads', 4, 'Number of BLAS threads')
cmd:option('-tensorType', 'torch.FloatTensor', 'Default tensor type')
cmd:option('-gpu', 1, 'GPU device ID (0 to disable)')
-- Game
cmd:option('-game', 'pong', 'Name of Atari ROM (stored in "roms" directory)')
-- Train vs. test mode
cmd:option('-mode', 'train', '"train" or "eval" mode')
-- Experience replay options
cmd:option('-memSize', 1000000, 'Experience replay memory size (# of tuples)')
cmd:option('-memSampleFreq', 4, 'Memory sample frequency')
cmd:option('-alpha', 1, 'Prioritised experience replay exponent α')
cmd:option('-betaZero', 1, 'Initial value of importance-sampling exponent β')
-- Reinforcement learning parameters
cmd:option('-gamma', 0.99, 'Discount rate γ')
cmd:option('-eta', 0.00007, 'Learning rate η') -- Accounts for prioritied experience sampling but not duel
cmd:option('-epsilonStart', 1, 'Initial value of greediness ε')
cmd:option('-epsilonEnd', 0.01, 'Final value of greediness ε')
cmd:option('-epsilonSteps', 1000000, 'Number of steps to linearly decay epsilonStart to epsilonEnd')
cmd:option('-tau', 30000, 'Steps between target net updates τ')
cmd:option('-rewardClamp', 1, 'Clamps reward magnitude')
cmd:option('-tdClamp', 1, 'Clamps TD-error δ magnitude')
-- Training options
cmd:option('-optimiser', 'rmsprop', 'Training algorithm')
cmd:option('-momentum', 0.95, 'SGD momentum')
cmd:option('-batchSize', 32, 'Minibatch size')
cmd:option('-steps', 50000000, 'Training iterations')
--cmd:option('-learnStart', 50000, 'Number of steps after which learning starts')
-- Evaluation options
cmd:option('-evalFreq', 1000000, 'Evaluation frequency')
cmd:option('-evalSize', 500, '# of validation transitions to use')
-- alewrap options
cmd:option('-actrep', 4, 'Times to repeat action')
cmd:option('-random_starts', 30, 'Play noop action between 1 and random_starts number of times at the start of each training episode')
-- TODO: Tidy up options/check agent_params
--cmd:option('-agent_params', 'hist_len=4,update_freq=4,n_replay=1,ncols=1,bufferSize=512', 'string of agent parameters')
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
    opt.step = step -- Pass step to agent for use in training

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
