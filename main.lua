local Setup = require 'Setup'
local ExperienceReplay = require 'ExperienceReplay'

local setup = Setup(arg)
local opt = setup.opt

local experienceReplay = ExperienceReplay(opt)

if opt.mode == 'train' then

  experienceReplay:train()

elseif opt.mode == 'eval' then

  experienceReplay:evaluate()

end
