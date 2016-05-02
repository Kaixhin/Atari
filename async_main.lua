require 'logroll'
local AsyncMaster = require 'AsyncMaster'

local cmd = torch.CmdLine()
-- Base Torch7 options
cmd:option('-seed', 1, 'Random seed')
cmd:option('-threads', 4, 'Number of async threads')
cmd:option('-tensorType', 'torch.FloatTensor', 'Default tensor type')
cmd:option('-game', 'catch', 'Name of Atari ROM (stored in "roms" directory)') -- Uses "Catch" env by default
cmd:option('-mode', 'train', 'Train vs. test mode: train|eval')
-- Screen preprocessing options
cmd:option('-height', 84, 'Resized screen height')
cmd:option('-width', 84, 'Resize screen width')
cmd:option('-colorSpace', 'y', 'Colour space conversion (screen is RGB): rgb|y|lab|yuv|hsl|hsv|nrgb')
-- Agent options
cmd:option('-histLen', 4, 'Number of consecutive states processed')
cmd:option('-duel', 'true', 'Use dueling network architecture (learns advantage function)')
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
cmd:option('-eta', 0.0000625, 'Learning rate η') -- Prioritied experience replay learning rate (1/4 that of DQN; does not account for Duel as well)
cmd:option('-momentum', 0.95, 'Gradient descent momentum')
cmd:option('-batchSize', 5, 'Minibatch size')
cmd:option('-steps', 5e7, 'Training iterations (steps)') -- Frame := step in ALE; Time step := consecutive frames treated atomically by the agent
cmd:option('-gradClip', 10, 'Clips L2 norm of gradients at gradClip (0 to disable)')
-- Evaluation options
cmd:option('-progFreq', 10000, 'Interval of steps between reporting progress')
cmd:option('-reportWeights', 'false', 'Report weight and weight gradient statistics')
cmd:option('-valFreq', 250000, 'Interval of steps between validating agent') -- valFreq steps is used as an epoch, hence #epochs = steps/valFreq
cmd:option('-valSteps', 125000, 'Number of steps to use for validation')
cmd:option('-valSize', 500, 'Number of transitions to use for calculating validation statistics')
-- ALEWrap options
cmd:option('-fullActions', 'false', 'Use full set of 18 actions')
cmd:option('-actRep', 4, 'Times to repeat action') -- Independent of history length
cmd:option('-randomStarts', 30, 'Max number of no-op actions played before presenting the start of each training episode')
cmd:option('-poolFrmsType', 'max', 'Type of pooling over previous emulator frames: max|mean')
cmd:option('-poolFrmsSize', 2, 'Number of emulator frames to pool over')
-- Experiment options
cmd:option('-_id', '', 'ID of experiment (used to store saved results, defaults to game name)')
cmd:option('-network', '', 'Saved network weights file to load (weights.t7)')
cmd:option('-verbose', 'false', 'Log info for every episode (only in train mode)')
-- Async
cmd:option('-async', '1stepq', 'async method')
local opt = cmd:parse(arg)

-- Process boolean options (Torch fails to accept false on the command line)
opt.duel = opt.duel == 'true' or false
opt.doubleQ = opt.doubleQ == 'true' or false
opt.reportWeights = opt.reportWeights == 'true' or false
opt.fullActions = opt.fullActions == 'true' or false
opt.verbose = opt.verbose == 'true' or false
opt.record = opt.record == 'true' or false
opt.bootstraps = 0
opt.gpu = 0
opt.ale = opt.game ~= 'catch'

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

-- Check enough validation transitions will be collected before first validation
if opt.valFreq <= opt.valSize then
  log.error('valFreq must be greater than valSize')
  error('valFreq must be greater than valSize')
end

-- Torch setup
log.info('Setting up Torch7')
-- Use enhanced garbage collector
torch.setheaptracking(true)
-- Set number of BLAS threads
torch.setnumthreads(1)
-- Set default Tensor type (float is more efficient than double)
torch.setdefaulttensortype(opt.tensorType)
-- Set manual seed
torch.manualSeed(opt.seed)

-- Tensor creation function for removing need to cast to CUDA if GPU is enabled
opt.Tensor = function(...)
  return torch.Tensor(...)
end


local master = AsyncMaster(opt)

master:start()

