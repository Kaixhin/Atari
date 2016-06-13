local classic = require 'classic'
local Model = require 'Model'

local AsyncModel = classic.class('AsyncModel')

function AsyncModel:_init(opt)
  log.info('Setting up ' .. (opt.ale and 'Arcade Learning Environment' or 'Catch'))

  if opt.ale then
    local Atari = require 'rlenvs.Atari'
    self.env = Atari(opt)
    local stateSpec = self.env:getStateSpec()

    -- Provide original channels, height and width for resizing from
    opt.origChannels, opt.origHeight, opt.origWidth = table.unpack(stateSpec[2])
  else
    local Catch = require 'rlenvs.Catch'
    self.env = Catch()
    local stateSpec = self.env:getStateSpec()
    
    -- Provide original channels, height and width for resizing from
    opt.origChannels, opt.origHeight, opt.origWidth = table.unpack(stateSpec[2])

    -- Adjust height and width
    opt.height, opt.width = stateSpec[2][2], stateSpec[2][3]
  end

  self.model = Model(opt)
  self.a3c = opt.async == 'A3C'

  classic.strict(self)
end

function AsyncModel:getEnvAndModel()
  return self.env, self.model
end

function AsyncModel:createNet()
  local actionSpec = self.env:getActionSpec()
  local m = actionSpec[3][2] - actionSpec[3][1] + 1 -- Number of discrete actions
  return self.model:create(m)
end


return AsyncModel