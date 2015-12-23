-- Parameters taken from Double DQN specs
require 'cutorch'
local image = require 'image'
local optim = require 'optim'
local environment = require 'environment'
local model = require 'model'

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
cmd:option('-mode', 'train', '"train" or "test" mode')
-- Experience replay options
cmd:option('-expReplMem', 1000000, 'Experience replay memory (# of tuples)')
cmd:option('-memSampleFreq', 4, 'Memory sample frequency')
-- TODO: Add prioritised experience replay
-- Reinforcement learning parameters
cmd:option('-gamma', 0.99, 'Discount rate γ')
cmd:option('-alpha', 0.00025, 'Learning rate α')
cmd:option('-epsilonStart', 1, 'Initial value of greediness ε')
cmd:option('-epsilonEnd', 0.1, 'Final value of greediness ε')
cmd:option('-epsilonSteps', 1000000, 'Number of steps to linearly decay epsilonStart to epsilonEnd')
cmd:option('-tau', 10000, 'Steps between target net updates τ')
cmd:option('-tdClamp', 1, 'Clamps TD error magnitude')
-- Training options
cmd:option('-optimiser', 'rmsprop', 'Training algorithm')
cmd:option('-momentum', 0.95, 'SGD momentum')
cmd:option('-batchSize', 32, 'Minibatch size')
cmd:option('-steps', 50000000, 'Training iterations')
cmd:option('-learnStart', 50000, 'Number of steps after which learning starts')
-- alewrap options
cmd:option('-actrep', 4, 'Times to repeat action')
cmd:option('-random_starts', 30, 'Play action 0 between 1 and random_starts number of times at the start of each training episode')
-- TODO: Tidy up options/check agent_params
--[[
cmd:option('-eval_freq', 250000, 'Frequency of greedy evaluation')
cmd:option('-eval_steps', 125000, 'Number of evaluation steps')
cmd:option('-prog_freq', 5000, 'Frequency of progress output')
cmd:option('-save_freq', 125000, 'Frequency of saving model')
cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '<snapshot filename>', 'reload pretrained network')
cmd:option('-agent_params', 'hist_len=4,update_freq=4,n_replay=1,ncols=1,bufferSize=512,valid_size=500,min_reward=-1,max_reward=1', 'string of agent parameters')
--]]
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
cutorch.setDevice(opt.gpu)
cutorch.manualSeedAll(torch.random())
-- TODO: Confirm necessary to override print to always flush the output
--[[
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end
--]]

-- TODO: Separate out options so that this is unneeded
-- Converts strings to tables
local function str_to_table(str)
  if type(str) == 'table' then
    return str
  end
  if not str or type(str) ~= 'string' then
    if type(str) == 'table' then
      return str
    end
    return {}
  end
  local ttr
  if str ~= '' then
    local ttx=tt
    loadstring('tt = {' .. str .. '}')()
    ttr = tt
    tt = ttx
  else
    ttr = {}
  end
  return ttr
end
if opt.agent_params then
  opt.agent_params = str_to_table(opt.agent_params)
end

-- Initialise Arcade Learning Environment
local gameEnv = environment.init(opt)
local A = gameEnv:getActions() -- Set of actions

-- Create agent
local agent = model.createAgent(gameEnv, opt)

-- TODO: Sort out variables
--[[
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {0}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local total_reward
local nrewards
local nepisodes
local episode_reward
--]]

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

  for step = 1, opt.steps do
    opt.step = step -- Pass step to agent for training
    --local action_index = agent:perceive(reward, screen, terminal)
    local actionIndex = agent:observe(screen, 'train') -- TODO: Add terminal?
    if not terminal then
      screen, reward, terminal = gameEnv:step(A[actionIndex], true) -- True flag for training mode
      agent:learn(reward)
    else
      if opt.random_starts > 0 then
        screen, reward, terminal = gameEnv:nextRandomGame()
      else
        screen, reward, terminal = gameEnv:newGame()
      end
    end

    if qt then
      image.display({image=screen, win=window})
    end
  end
elseif opt.mode == 'test' then
  -- Play one game (episode)
  while not terminal do
    -- if action was chosen randomly, Q-value is 0
    --agent.bestq = 0
    local actionIndex = agent:osberve(screen, 'test')
    screen, reward, terminal = gameEnv:step(A[actionIndex], false) -- Flag for test mode

    if qt then
      image.display({image=screen, win=window})
    end
  end
end

-- Training
--[[
local optimConfig = {
  learningRate = alpha,
  alpha = momentum
}
optim[optimiser](func, x, optimConfig)
--]]
