local Setup = require 'Setup'
local Master = require 'Master'
local AsyncMaster = require 'async/AsyncMaster'

-- Parse options and perform setup
local setup = Setup(arg)
local opt = setup.opt

-- Start master experiment runner
if opt.async then
  local master = AsyncMaster(opt)

  master:start() -- TODO: Use same API as normal master
else
  local master = Master(opt)

  if opt.mode == 'train' then
    master:train()
  elseif opt.mode == 'eval' then
    master:evaluate()
  end
end
