local classic = require 'classic'
local Model = require 'Model'

local AsyncModel = classic.class('AsyncModel')

function AsyncModel:_init(opt)
  -- Initialise environment
  log.info('Setting up ' .. opt.env)
  local Env = require(opt.env)
  self.env = Env(opt) -- Environment instantiation

  -- Set up fake training mode (if needed)
  if not self.env.training then
    self.env.training = function() end
  end
  -- Set up fake evaluation mode (if needed)
  if not self.env.evaluate then
    self.env.evaluate = function() end
  end
  -- Set up fake display (if needed)
  if not self.env.getDisplay then
    self.env.getDisplay = function() end -- TODO: Implement for Atari and Catch
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
