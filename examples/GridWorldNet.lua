local nn = require 'nn'
require 'classic.torch' -- Enables serialisation

local Body = classic.class('Body')

-- Constructor
function Body:_init(opts)
  opts = opts or {}
end

function Body:createBody()
  local net = nn.Sequential()
  net:add(nn.View(2))
  net:add(nn.Linear(2, 32))
  net:add(nn.ReLU(true))

  return net
end

return Body
