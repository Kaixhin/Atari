local classic = require 'classic'
local optim = require 'optim'
require 'modules/sharedRmsProp'

local A3CAgent = classic.class('A3CAgent')

function A3CAgent:_init(opt)
  log.info('creating QAgent')
  local asyncModel = AsyncModel(opt)
  self.env, self.model = asyncModel:getEnvAndModel()

  self.id = __threadid or 1
  self.counters = counters

  self.optimiser = optim[opt.optimiser]
  self.optimParams = {
    learningRate = opt.eta,
    momentum = opt.momentum,
    g = sharedG
  }

  self.learningRateStart = opt.eta

  local actionSpec = self.env:getActionSpec()
  self.m = actionSpec[3][2] - actionSpec[3][1] + 1
  self.actionOffset = 1 - actionSpec[3][1]

  self.policyNet = policyNet:clone('weight', 'bias')
  self.targetNet = targetNet:clone('weight', 'bias')
  self.targetNet:evaluate()

  self.theta = theta
  local __, gradParams = self.policyNet:parameters()
  self.dTheta = nn.Module.flatten(gradParams)
  self.dTheta:zero()

  self.ale = opt.ale

  self.stateBuffer = CircularQueue(opt.histLen, opt.Tensor, {opt.nChannels, opt.height, opt.width})

  self.gamma = opt.gamma -- ???
  self.rewardClip = opt.rewardClip
  self.tdClip = opt.tdClip
  self.gradClip = opt.gradClip

  self.progFreq = opt.progFreq
  self.batchSize = opt.batchSize

  self.Tensor = opt.Tensor

  self.batchIdx = 0
  self.target = self.Tensor(self.m)

  self.totalSteps = math.floor(opt.steps / opt.threads)

  self.tic = 0
  self.step = 0

  classic.strict(self)
end


function A3CAgent:learn(steps)

end


