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
cmd:option('-epsilon', 1, 'Greediness ε (decreases linearly from 1 to 0.1 over expReplMem steps)') -- TODO: Parameterise decay
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
  for step = 1, opt.steps do
    --local action_index = agent:perceive(reward, screen, terminal)
    local actionIndex = agent:act(screen) -- TODO: Add terminal?
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

    --[[
    if step % opt.prog_freq == 0 then
        assert(step==agent.numSteps, 'trainer step: ' .. step ..
                ' & agent.numSteps: ' .. agent.numSteps)
        print("Steps: ", step)
        agent:report()
    end
    --]]

    --[[
    if step % opt.eval_freq == 0 and step > opt.learnStart then
      screen, reward, terminal = gameEnv:newGame()

      total_reward = 0
      nrewards = 0
      nepisodes = 0
      episode_reward = 0

      local eval_time = sys.clock()
      for estep = 1, opt.eval_steps do
        --local action_index = agent:perceive(reward, screen, terminal, true, 0.05)
        local actionIndex = agent:act(screen)

        -- Play game in test mode (episodes don't end when losing a life)
        screen, reward, terminal = gameEnv:step(A[actionIndex], false)

        -- record every reward
        episode_reward = episode_reward + reward
        if reward ~= 0 then
          nrewards = nrewards + 1
        end

        if terminal then
          total_reward = total_reward + episode_reward
          episode_reward = 0
          nepisodes = nepisodes + 1
          screen, reward, terminal = gameEnv:nextRandomGame()
        end
      end

      eval_time = sys.clock() - eval_time
      start_time = start_time + eval_time
      agent:compute_validation_statistics()
      local ind = #reward_history+1
      total_reward = total_reward/math.max(1, nepisodes)

      if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
        agent.best_network = agent.network:clone()
      end

      if agent.v_avg then
        v_history[ind] = agent.v_avg
        td_history[ind] = agent.tderr_avg
        qmax_history[ind] = agent.q_max
      end
      print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

      reward_history[ind] = total_reward
      reward_counts[ind] = nrewards
      episode_counts[ind] = nepisodes

      time_history[ind+1] = sys.clock() - start_time

      local time_dif = time_history[ind+1] - time_history[ind]

      local training_rate = opt.actrep*opt.eval_freq/time_dif
    end
    --]]

    -- TODO: Saving network
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
    
    local actionIndex = agent:act(screen)
    screen, reward, terminal = gameEnv:step(A[actionIndex], false) -- Flag for test mode

    if qt then
      image.display({image=screen, win=window})
    end
  end
end

--[[

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
