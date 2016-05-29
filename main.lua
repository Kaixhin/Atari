local Setup = require 'Setup'
local ExperienceReplay = require 'ExperienceReplay'
local AsyncMaster = require 'async/AsyncMaster'

local setup = Setup(arg)
local opt = setup.opt

if opt.async then
  log.info(opt)
  local master = AsyncMaster(opt)
  master:start()

else
  local experienceReplay = ExperienceReplay(opt)

  if opt.mode == 'train' then
    experienceReplay:train()

  elseif opt.mode == 'eval' then
    experienceReplay:evaluate()

  end

end