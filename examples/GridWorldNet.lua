local nn = require 'nn'

local net = nn.Sequential()
net:add(nn.View(2))
net:add(nn.Linear(2, 32))
net:add(nn.ReLU(true))

return net
