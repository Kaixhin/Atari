local framework = require 'alewrap'

local environment = {}

-- Initialises ALE with game
environment.init = function(opt)
  -- Set GPU flag (GPU enables faster screen buffer)
  local gpu = opt.gpu - 1
  -- Create options from opt
  local options = {
    game_path = "roms",
    env = opt.game,
    actrep = opt.actrep,
    random_starts = opt.random_starts,
    gpu = gpu,
    pool_frms = {
      size = 2, -- Pool over frames to prevent problems with fixed interval events
      type = 'max' -- Defaults to 2-frame mean-pooling
    }
  }
  return framework.GameEnvironment(options)
end

return environment
