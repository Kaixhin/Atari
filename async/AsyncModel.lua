local classic = require 'classic'
local Model = require 'Model'

local AsyncModel = classic.class('AsyncModel')

function AsyncModel:_init(opt)
  -- Initialise environment
  log.info('Setting up ' .. opt.env)
  local Env = require(opt.env)
  self.env = Env(opt) -- Environment instantiation

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
