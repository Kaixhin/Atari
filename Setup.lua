----- General Setup -----
require 'logroll'
local cjson = require 'cjson'
local classic = require 'classic'
local _ = require 'moses'

local Setup = classic.class('Setup')

function Setup:_init(arg)
  -- Create log10 for Lua 5.2
  if not math.log10 then
    math.log10 = function(x)
      return math.log(x, 10)
    end
  end

  local opt = self:options(arg)

  -- Set up logs
  local flog = logroll.file_logger(paths.concat(opt.experiments, opt._id, 'log.txt'))
  local plog = logroll.print_logger()
  log = logroll.combine(flog, plog)

  self:validateOptions(opt)

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

  self.opt = opt
  classic.strict(self)
end


function Setup:options(arg)
  -- Detect and use GPU 1 by default
  local cuda = pcall(require, 'cutorch')

  local cmd = torch.CmdLine()
  -- Base Torch7 options
  cmd:option('-seed', 1, 'Random seed')
  cmd:option('-threads', 4, 'Number of BLAS or async threads')
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
  -- Model options
  cmd:option('-hiddenSize', 512, 'Number of units in the hidden fully connected layer')
  cmd:option('-histLen', 4, 'Number of consecutive states processed/used for backpropagation-through-time') -- DQN standard is 4, DRQN is 10
  cmd:option('-duel', 'true', 'Use dueling network architecture (learns advantage function)')
  cmd:option('-bootstraps', 10, 'Number of bootstrap heads (0 to disable)')
  --cmd:option('-bootstrapMask', 1, 'Independent probability of masking a transition for each bootstrap head ~ Ber(bootstrapMask) (1 to disable)')
  cmd:option('-recurrent', 'false', 'Use recurrent connections')
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
  cmd:option('-eta', 0.0000625, 'Learning rate η') -- Prioritied experience replay learning rate (1/4 that of DQN; does not account for Duel as well)
  cmd:option('-momentum', 0.95, 'Gradient descent momentum')
  cmd:option('-batchSize', 32, 'Minibatch size')
  cmd:option('-steps', 5e7, 'Training iterations (steps)') -- Frame := step in ALE; Time step := consecutive frames treated atomically by the agent
  cmd:option('-learnStart', 50000, 'Number of steps after which learning starts')
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
  -- Async options
  cmd:option('-async', 'false', 'async method') -- OneStepQ|NStepQ|Sarsa|A3C
  cmd:option('-rmsEpsilon', 0.1, 'Epsilon for sharedRmsProp')
  cmd:option('-novalidation', 'false', 'dont run validation thread in async') -- for debugging
  -- Experiment options
  cmd:option('-experiments', 'experiments', 'Base directory to store experiments')
  cmd:option('-_id', '', 'ID of experiment (used to store saved results, defaults to game name)')
  cmd:option('-network', '', 'Saved network weights file to load (weights.t7)')
  cmd:option('-verbose', 'false', 'Log info for every episode (only in train mode)')
  cmd:option('-saliency', 'none', 'Display saliency maps (requires QT): none|normal|guided|deconvnet')
  cmd:option('-record', 'false', 'Record screen (only in eval mode)')
  local opt = cmd:parse(arg)

  -- Process boolean options (Torch fails to accept false on the command line)
  opt.duel = opt.duel == 'true'
  opt.recurrent = opt.recurrent == 'true'
  opt.doubleQ = opt.doubleQ == 'true'
  opt.reportWeights = opt.reportWeights == 'true'
  opt.fullActions = opt.fullActions == 'true'
  opt.verbose = opt.verbose == 'true'
  opt.record = opt.record == 'true'
  opt.novalidation = opt.novalidation == 'true'
  if opt.async == 'false' then opt.async = false end
  if opt.async then opt.gpu = 0 end

  -- Set ID as game name if not set
  if opt._id == '' then
    opt._id = opt.game
  end

  opt.ale = opt.game ~= 'catch'

  -- Create experiment directory
  if not paths.dirp(opt.experiments) then
    paths.mkdir(opt.experiments)
  end
  paths.mkdir(paths.concat(opt.experiments, opt._id))
  -- Save options for reference
  local file = torch.DiskFile(paths.concat(opt.experiments, opt._id, 'opts.json'), 'w')
  file:writeString(cjson.encode(opt))
  file:close()

  return opt
end


local function abortIf(notOk, err)
  if notOk then 
    log.error(err)
    error(err)
  end
end


function Setup:validateOptions(opt)
  -- Calculate number of colour channels
  abortIf(not _.contains({'rgb', 'y', 'lab', 'yuv', 'hsl', 'hsv', 'nrgb'}, opt.colorSpace),
    'Unsupported colour space for conversion')
  opt.nChannels = opt.colorSpace == 'y' and 1 or 3

  -- Check start of learning occurs after at least one minibatch of data has been collected
  abortIf(opt.learnStart <= opt.batchSize, 'learnStart must be greater than batchSize')

  -- Check enough validation transitions will be collected before first validation
  abortIf(opt.valFreq <= opt.valSize, 'valFreq must be greater than valSize')

  -- Check prioritised experience replay options
  abortIf(not _.contains({'none', 'rank', 'proportional'}, opt.memPriority),
    'Type of prioritised experience replay unrecognised')

  -- Check start of learning occurs after at least 1/100 of memory has been filled
  abortIf(opt.learnStart <= opt.memSize/100, 'learnStart must be greater than memSize/100')

  -- Check memory size is multiple of 100 (makes prioritised sampling partitioning simpler)
  abortIf(opt.memSize % 100 ~= 0, 'memSize must be a multiple of 100')

  -- Check learning occurs after first progress report
  abortIf(opt.learnStart < opt.progFreq, 'learnStart must be greater than progFreq')

  -- Check saliency map options
  abortIf(not _.contains({'none', 'normal', 'guided', 'deconvnet'}, opt.saliency),
    'Unrecognised method for visualising saliency maps')

  if opt.async then
    abortIf(opt.recurrent and opt.async ~= 'OneStepQ', 'recurrent only supported for OneStepQ in async for now')
    abortIf(opt.PALpha > 0, 'PAL not supported in async modes yet')
    abortIf(opt.bootstraps > 0, 'bootstraps not supported in async mode')
    abortIf(opt.async == 'A3C' and opt.duel, 'dueling and A3C dont mix')
    abortIf(opt.async == 'A3C' and opt.doubleQ, 'doubleQ and A3C dont mix')
  end
end


return Setup
