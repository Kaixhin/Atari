local classic = require 'classic'
local Model = require 'Model'

local AsyncModel = classic.class('AsyncModel')

function AsyncModel:_init(opt)
  -- Initialise environment
  log.info('Setting up ' .. opt.env)
  local Env = require(opt.env)
  self.env = Env(opt) -- Environment instantiation

  -- Augment environment with extra methods if missing
  if not self.env.training then
    self.env.training = function() end
  end
  if not self.env.evaluate then
    self.env.evaluate = function() end
  end

  self.model = Model(opt)
  self.a3c = opt.async == 'A3C'

  classic.strict(self)
end

function AsyncModel:getEnvAndModel()
  return self.env, self.model
end

function AsyncModel:createNet()
  return self.model:create()
end

return AsyncModel
