local Display = require 'Display'
local ValidationAgent = require 'async/ValidationAgent'
local AsyncModel = require 'async/AsyncModel'
local classic = require 'classic'
local tds = require 'tds'

local AsyncEvaluation = classic.class('AsyncEvaluation')


function AsyncEvaluation:_init(opt)
  local asyncModel = AsyncModel(opt)
  local env = asyncModel:getEnvAndModel()
  local policyNet = asyncModel:createNet()
  local theta = policyNet:getParameters()

  local weightsFile = paths.concat('experiments', opt._id, 'last.weights.t7')
  local weights = torch.load(weightsFile)
  theta:copy(weights)

  local atomic = tds.AtomicCounter()
  self.validAgent = ValidationAgent(opt, theta, atomic)

  local state = env:start()
  self.hasDisplay = false
  if opt.displaySpec then
    self.hasDisplay = true
    self.display = Display(opt, env:getDisplay())
  end

  classic.strict(self)
end


function AsyncEvaluation:evaluate()
  local display = self.hasDisplay and self.display or nil
  self.validAgent:evaluate(display)
end

return AsyncEvaluation
