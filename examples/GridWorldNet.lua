local nn = require 'nn'
local nninit = require 'nninit'
require 'classic.torch' -- Enables serialisation
--require 'rnn'
--require 'dpnn' -- Adds gradParamClip method

local Body = classic.class('Body')

-- Constructor
function Body:_init(opts)
  opts = opts or {}

  --self.recurrent = opts.recurrent
  --self.histLen = opts.histLen
  --self.stateSpec = opts.stateSpec
end

function Body:createBody()
  local net

  net = nn.Sequential()
  net:add(nn.View(2))
  net:add(nn.Linear(2, 32))
  net:add(nn.ReLU(true))

  return net
end

return Body
