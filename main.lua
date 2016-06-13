local Setup = require 'Setup'
local ExperienceReplay = require 'ExperienceReplay'
local AsyncMaster = require 'async/AsyncMaster'
local AsyncEvaluation = require 'async/AsyncEvaluation'

local setup = Setup(arg)
local opt = setup.opt

if opt.async then
  log.info(opt)

  if opt.mode == 'train' then
    local master = AsyncMaster(opt)
    master:start()
  elseif opt.mode == 'eval' then
    local eval = AsyncEvaluation(opt)
    eval:evaluate()
  end
else
  local experienceReplay = ExperienceReplay(opt)

  if opt.mode == 'train' then
    experienceReplay:train()
  elseif opt.mode == 'eval' then
    experienceReplay:evaluate()
  end
end
