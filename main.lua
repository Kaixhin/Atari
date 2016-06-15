local Setup = require 'Setup'
local Master = require 'Master'
local AsyncMaster = require 'async/AsyncMaster'
local AsyncEvaluation = require 'async/AsyncEvaluation'

-- Parse options and perform setup
local setup = Setup(arg)
local opt = setup.opt

-- Start master experiment runner
if opt.async then
  if opt.mode == 'train' then
    local master = AsyncMaster(opt)
    master:start()
  elseif opt.mode == 'eval' then
    local eval = AsyncEvaluation(opt)
    eval:evaluate()
  end
else
  local master = Master(opt)

  if opt.mode == 'train' then
    master:train()
  elseif opt.mode == 'eval' then
    master:evaluate()
  end
end
