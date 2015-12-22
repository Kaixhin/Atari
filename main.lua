-- Parameters taken from Double DQN specs
require 'cutorch'
local image = require 'image'
local optim = require 'optim'
local environment = require 'environment'
local model = require 'model'

local cmd = torch.CmdLine()
cmd:option('-seed', 123, 'Random seed')
cmd:option('-threads', 4, 'Number of BLAS threads')
cmd:option('-tensorType', 'torch.FloatTensor', 'Default tensor type')
cmd:option('-gpu', 1, 'GPU device ID (0 to disable)')
cmd:option('-game', 'pong', 'Name of Atari ROM (stored in "roms" directory)')
cmd:option('-mode', 'train', '"train" or "test" mode')
cmd:option('-gamma', 0.99, 'Discount rate γ')
cmd:option('-alpha', 0.00025, 'Learning rate α')
cmd:option('-epsilon', 1, 'Greediness ε (decreases linearly from 1 to 0.1 over 1M steps)')
cmd:option('-tau', 10000, 'Steps between target net updates τ')
cmd:option('-expReplMem', 1000000, 'Experience replay memory (# of tuples)')
cmd:option('-memSampleFreq', 4, 'Memory sample frequency')
cmd:option('-optimiser', 'rmsprop', 'Training algorithm')
cmd:option('-momentum', 0.95, 'SGD momentum')
cmd:option('-batchSize', 32, 'Minibatch size')
cmd:option('-nIterations', 50000000, 'Training iterations')
cmd:option('-validationFreq', 1000000, 'Validation frequency')

cmd:option('-actrep', 4, 'how many times to repeat action') -- TODO: 1 for training?
cmd:option('-random_starts', 30, 'play action 0 between 1 and random_starts number of times at the start of each training episode') -- TODO: 0 for training?
cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '<snapshot filename>', 'reload pretrained network')
cmd:option('-agent', 'NeuralQLearner', 'name of agent file to use')
cmd:option('-agent_params', 'hist_len=4,learn_start=50000,update_freq=4,n_replay=1,ncols=1,bufferSize=512,valid_size=500,clip_delta=1,min_reward=-1,max_reward=1', 'string of agent parameters')
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
local gameEnv = environment.init(opt.game)
local gameActions = gameEnv:getActions()
print(gameActions)

-- Creates agent
local agent = model.createAgent(gameActions)

local screen, reward, terminal = gameEnv:newGame()
local window = image.display({image=screen})

if opt.mode == 'train' then
  local iter = 0
  while iter < opt.nIterations do
    iter = iter + 1

  end
elseif opt.mode == 'test' then
  -- Play one game (episode)
  while not terminal do
    -- if action was chosen randomly, Q-value is 0
    --agent.bestq = 0
    -- choose the best action
    --local action_index = agent:perceive(reward, screen, terminal, true, 0.05)
    -- play game in test mode (episodes don't end when losing a life)
    --screen, reward, terminal = game_env:step(game_actions[action_index], false)
    
    local actionIndex = agent:observe(screen)
    screen, reward, terminal = gameEnv:step(gameActions[actionIndex], false)

    -- Update screen
    image.display({image=screen, win=window})
  end
end

--[[
-- Create agent
local agent = model.createAgent()
-- Model parameters θ
local theta, dTheta = agent:getParameters()

-- Reinforcement learning variables
local s = environment:start()
local a, sPrime, r
local isTerminal = false
while not isTerminal do
  -- Process the current state to pick an action
  local output = agent:forward(s)
  __, a = torch.max(output, 1)
  a = a[1]

  -- Perform a step in the environment
  sPrime, r, isTerminal = environment.step(s, a)

  -- Calculate max Q-value from next step
  local outputPrime = agent:forward(sPrime)
  local QPrime = torch.max(outputPrime)
  -- Calculate target Y
  local Y = r + gamma*QPrime
  -- Calculate error
  local err = torch.mul(theta, (Y - output[a]))

  -- TODO: Save experience

  -- Replace s with s'
  s = sPrime
end
--]]
-- Training
--[[
local optimConfig = {
  learningRate = alpha,
  alpha = momentum
}
optim[optimiser](func, x, optimConfig)
--]]
