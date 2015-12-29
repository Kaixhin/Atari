local framework = require 'alewrap'

local environment = {}

-- Initialises ALE with game
environment.init = function(opt)
  -- Set GPU flag (GPU enables faster screen buffer with CudaTensors)
  local gpu = opt.gpu - 1
  -- Create options from opt
  local options = {
    game_path = "roms",
    env = opt.game,
    actrep = opt.actRep,
    random_starts = opt.randomStarts,
    gpu = gpu,
    pool_frms = { -- Defaults to 2-frame mean-pooling
      type = opt.poolFrmsType,
      size = opt.poolFrmsSize -- Pools over frames to prevent problems with fixed interval events like lasers blinking
    }
  }
  -- Return game environment
  return framework.GameEnvironment(options)
end

return environment
