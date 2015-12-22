local framework = require 'alewrap'

local environment = {}

-- Initialises ALE with game
environment.init = function(game)
  return framework.GameEnvironment({game_path="roms", env=game})
end

return environment
