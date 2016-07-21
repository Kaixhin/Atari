require 'logroll'
local _ = require 'moses'
local classic = require 'classic'
local cjson = require 'cjson'

local Setup = classic.class('Setup')

-- Performs global setup
function Setup:_init(arg)
  -- Create log10 for Lua 5.2
  if not math.log10 then
    math.log10 = function(x)
      return math.log(x, 10)
    end
  end

  -- Parse command-line options
  self.opt = self:parseOptions(arg)

  -- Create experiment directory
  if not paths.dirp(self.opt.experiments) then
    paths.mkdir(self.opt.experiments)
  end
  paths.mkdir(paths.concat(self.opt.experiments, self.opt._id))
  -- Save options for reference
  local file = torch.DiskFile(paths.concat(self.opt.experiments, self.opt._id, 'opts.json'), 'w')
  file:writeString(cjson.encode(self.opt))
  file:close()

  -- Set up logging
  local flog = logroll.file_logger(paths.concat(self.opt.experiments, self.opt._id, 'log.txt'))
  local plog = logroll.print_logger()
  log = logroll.combine(flog, plog) -- Global logger

  -- Validate command-line options (logging errors)
  self:validateOptions()

  -- Augment environments to meet spec
  self:augmentEnv()

  -- Torch setup
  log.info('Setting up Torch7')
  -- Use enhanced garbage collector
  torch.setheaptracking(true)
  -- Set number of BLAS threads
  torch.setnumthreads(self.opt.threads)
  -- Set default Tensor type (float is more efficient than double)
  torch.setdefaulttensortype(self.opt.tensorType)
  -- Set manual seed
  torch.manualSeed(self.opt.seed)

  -- Tensor creation function for removing need to cast to CUDA if GPU is enabled
  -- TODO: Replace with local functions across codebase
  self.opt.Tensor = function(...)
    return torch.Tensor(...)
  end

  -- GPU setup
  if self.opt.gpu > 0 then
    log.info('Setting up GPU')
    cutorch.setDevice(self.opt.gpu)
    -- Set manual seeds using random numbers to reduce correlations
    cutorch.manualSeed(torch.random())
    -- Replace tensor creation function
    self.opt.Tensor = function(...)
      return torch.CudaTensor(...)
    end
  end

  classic.strict(self)
end

-- Parses command-line options
function Setup:parseOptions(arg)
  -- Detect and use GPU 1 by default
  local cuda = pcall(require, 'cutorch')

  local cmd = torch.CmdLine()
  -- Base Torch7 options
  cmd:option('-seed', 1, 'Random seed')
  cmd:option('-threads', 4, 'Number of BLAS or async threads')
  cmd:option('-tensorType', 'torch.FloatTensor', 'Default tensor type')
  cmd:option('-gpu', cuda and 1 or 0, 'GPU device ID (0 to disable)')
  -- Environment options
  cmd:option('-env', 'rlenvs.Catch', 'Environment class (Lua file to be loaded/rlenv)')
  cmd:option('-zoom', 1, 'Display zoom (requires QT)')
  cmd:option('-game', '', 'Name of Atari ROM (stored in "roms" directory)')
  -- Training vs. evaluate mode
  cmd:option('-mode', 'train', 'Train vs. test mode: train|eval')
  -- State preprocessing options (for visual states)
  cmd:option('-height', 0, 'Resized screen height (0 to disable)')
  cmd:option('-width', 0, 'Resize screen width (0 to disable)')
  cmd:option('-colorSpace', '', 'Colour space conversion (screen is RGB): <none>|y|lab|yuv|hsl|hsv|nrgb')
  -- Model options
  cmd:option('-modelBody', '', 'Path to Torch nn model to be used as DQN "body"')
  cmd:option('-hiddenSize', 512, 'Number of units in the hidden fully connected layer')
  cmd:option('-histLen', 4, 'Number of consecutive states processed/used for backpropagation-through-time') -- DQN standard is 4, DRQN is 10
  cmd:option('-duel', 'true', 'Use dueling network architecture (learns advantage function)')
  cmd:option('-bootstraps', 10, 'Number of bootstrap heads (0 to disable)')
  --cmd:option('-bootstrapMask', 1, 'Independent probability of masking a transition for each bootstrap head ~ Ber(bootstrapMask) (1 to disable)')
  cmd:option('-recurrent', 'false', 'Use recurrent connections')
  -- Experience replay options
  cmd:option('-discretiseMem', 'true', 'Discretise states to integers ∈ [0, 255] for storage')
  cmd:option('-memSize', 1e6, 'Experience replay memory size (number of tuples)')
  cmd:option('-memSampleFreq', 4, 'Interval of steps between sampling from memory to learn')
  cmd:option('-memNSamples', 1, 'Number of times to sample per learning step')
  cmd:option('-memPriority', '', 'Type of prioritised experience replay: <none>|rank|proportional') -- TODO: Implement proportional prioritised experience replay
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
  cmd:option('-noValidation', 'false', 'Disable asynchronous agent validation thread') -- TODO: Make behaviour consistent across Master/AsyncMaster
  cmd:option('-valFreq', 250000, 'Interval of steps between validating agent') -- valFreq steps is used as an epoch, hence #epochs = steps/valFreq
  cmd:option('-valSteps', 125000, 'Number of steps to use for validation')
  cmd:option('-valSize', 500, 'Number of transitions to use for calculating validation statistics')
  -- Async options
  cmd:option('-async', '', 'Async agent: <none>|Sarsa|OneStepQ|NStepQ|A3C') -- TODO: Change names
  cmd:option('-rmsEpsilon', 0.1, 'Epsilon for sharedRmsProp')
  cmd:option('-entropyBeta', 0.01, 'Policy entropy regularisation β')
  -- ALEWrap options
  cmd:option('-fullActions', 'false', 'Use full set of 18 actions')
  cmd:option('-actRep', 4, 'Times to repeat action') -- Independent of history length
  cmd:option('-randomStarts', 30, 'Max number of no-op actions played before presenting the start of each training episode')
  cmd:option('-poolFrmsType', 'max', 'Type of pooling over previous emulator frames: max|mean')
  cmd:option('-poolFrmsSize', 2, 'Number of emulator frames to pool over')
  cmd:option('-lifeLossTerminal', 'true', 'Use life loss as terminal signal (training only)')
  cmd:option('-flickering', 0, 'Probability of screen flickering (Catch only)')
  -- Experiment options
  cmd:option('-experiments', 'experiments', 'Base directory to store experiments')
  cmd:option('-_id', '', 'ID of experiment (used to store saved results, defaults to game name)')
  cmd:option('-network', '', 'Saved network weights file to load (weights.t7)')
  cmd:option('-verbose', 'false', 'Log info for every episode (only in train mode)')
  cmd:option('-saliency', '', 'Display saliency maps (requires QT): <none>|normal|guided|deconvnet')
  cmd:option('-record', 'false', 'Record screen (only in eval mode)')
  local opt = cmd:parse(arg)

  -- Process boolean options (Torch fails to accept false on the command line)
  opt.duel = opt.duel == 'true'
  opt.recurrent = opt.recurrent == 'true'
  opt.discretiseMem = opt.discretiseMem == 'true'
  opt.doubleQ = opt.doubleQ == 'true'
  opt.reportWeights = opt.reportWeights == 'true'
  opt.fullActions = opt.fullActions == 'true'
  opt.lifeLossTerminal = opt.lifeLossTerminal == 'true'
  opt.verbose = opt.verbose == 'true'
  opt.record = opt.record == 'true'
  opt.noValidation = opt.noValidation == 'true'

  -- Process boolean/enum options
  if opt.colorSpace == '' then opt.colorSpace = false end
  if opt.memPriority == '' then opt.memPriority = false end
  if opt.async == '' then opt.async = false end
  if opt.saliency == '' then opt.saliency = false end
  if opt.async then opt.gpu = 0 end -- Asynchronous agents are CPU-only

  -- Set ID as env (plus game name) if not set
  if opt._id == '' then
    local envName = paths.basename(opt.env)
    if opt.game == '' then
      opt._id = envName
    else
      opt._id = envName .. '.' .. opt.game
    end
  end
  
  -- Create one environment to extract specifications
  local Env = require(opt.env)
  local env = Env(opt)
  opt.stateSpec = env:getStateSpec()
  opt.actionSpec = env:getActionSpec()
  -- Process display if available (can be used for saliency recordings even without QT)
  if env.getDisplay then
    opt.displaySpec = env:getDisplaySpec()
  end

  return opt
end

-- Logs and aborts on error
local function abortIf(err, msg)
  if err then 
    log.error(msg)
    error(msg)
  end
end

-- Validates setup options
function Setup:validateOptions()
  -- Check environment state is a single tensor
  abortIf(#self.opt.stateSpec ~= 3 or not _.isArray(self.opt.stateSpec[2]), 'Environment state is not a single tensor')
  
  -- Check environment has discrete actions
  abortIf(self.opt.actionSpec[1] ~= 'int' or self.opt.actionSpec[2] ~= 1, 'Environment does not have discrete actions')

  -- Change state spec if resizing
  if self.opt.height ~= 0 then 
    self.opt.stateSpec[2][2] = self.opt.height
  end
  if self.opt.width ~= 0 then 
    self.opt.stateSpec[2][3] = self.opt.width
  end

  -- Check colour conversions
  if self.opt.colorSpace then
    abortIf(not _.contains({'y', 'lab', 'yuv', 'hsl', 'hsv', 'nrgb'}, self.opt.colorSpace), 'Unsupported colour space for conversion')
    abortIf(self.opt.stateSpec[2][1] ~= 3, 'Original colour space must be RGB for conversion')
    -- Change state spec if converting from colour to greyscale
    if self.opt.colorSpace == 'y' then
      self.opt.stateSpec[2][1] = 1
    end
  end

  -- Check start of learning occurs after at least one minibatch of data has been collected
  abortIf(self.opt.learnStart <= self.opt.batchSize, 'learnStart must be greater than batchSize')

  -- Check enough validation transitions will be collected before first validation
  abortIf(self.opt.valFreq <= self.opt.valSize, 'valFreq must be greater than valSize')

  -- Check prioritised experience replay options
  abortIf(self.opt.memPriority and not _.contains({'rank', 'proportional'}, self.opt.memPriority), 'Type of prioritised experience replay unrecognised')
  abortIf(self.opt.memPriority == 'proportional', 'Proportional prioritised experience replay not implemented yet') -- TODO: Implement

  -- Check start of learning occurs after at least 1/100 of memory has been filled
  abortIf(self.opt.learnStart <= self.opt.memSize/100, 'learnStart must be greater than memSize/100')

  -- Check memory size is multiple of 100 (makes prioritised sampling partitioning simpler)
  abortIf(self.opt.memSize % 100 ~= 0, 'memSize must be a multiple of 100')

  -- Check learning occurs after first progress report
  abortIf(self.opt.learnStart < self.opt.progFreq, 'learnStart must be greater than progFreq')

  -- Check saliency map options
  abortIf(self.opt.saliency and not _.contains({'normal', 'guided', 'deconvnet'}, self.opt.saliency), 'Unrecognised method for visualising saliency maps')
  
  -- Check saliency is valid
  abortIf(self.opt.saliency and not self.opt.displaySpec, 'Saliency cannot be shown without env:getDisplay()')
  abortIf(self.opt.saliency and #self.opt.stateSpec[2] ~= 3 and (self.opt.stateSpec[2][1] ~= 3 or self.opt.stateSpec[2][1] ~= 1), 'Saliency cannot be shown without visual state')

  -- Check async options
  if self.opt.async then
    abortIf(self.opt.recurrent and self.opt.async ~= 'OneStepQ', 'Recurrent connections only supported for OneStepQ in async for now')
    abortIf(self.opt.PALpha > 0, 'Persistent advantage learning not supported in async modes yet')
    abortIf(self.opt.bootstraps > 0, 'Bootstrap heads not supported in async mode yet')
    abortIf(self.opt.async == 'A3C' and self.opt.duel, 'Dueling networks and A3C are incompatible')
    abortIf(self.opt.async == 'A3C' and self.opt.doubleQ, 'Double Q-learning and A3C are incompatible')
    abortIf(self.opt.saliency, 'Saliency maps not supported in async modes yet')
  end
end

-- Augments environments with extra methods if missing
function Setup:augmentEnv()
  local Env = require(self.opt.env)
  local env = Env(self.opt)

  -- Set up fake training mode (if needed)
  if not env.training then
    Env.training = function() end
  end
  -- Set up fake evaluation mode (if needed)
  if not env.evaluate then
    Env.evaluate = function() end
  end
end

return Setup
