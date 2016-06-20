local classic = require 'classic'
local GridWorld = require 'rlenvs.GridWorld'

local GridWorldVis, super = classic.class('GridWorldVis', GridWorld)

function GridWorldVis:_init(opts)
  super._init(self)

  -- Create screen
  self.screen = torch.Tensor(3, 21, 21):zero()
end

function GridWorldVis:getStateSpec()
  return {'real', {2}, {0, 1}}
end

function GridWorldVis:getDisplaySpec()
  return {'real', {3, 21, 21}, {0, 1}}
end

function GridWorldVis:getDisplay()
  return self.screen
end

function GridWorldVis:drawPixel(draw)
  if draw then
    self.screen[{{}, {20*self.position[2]+1}, {20*self.position[1]+1}}] = 1
  else
    self.screen[{{}, {20*self.position[2]+1}, {20*self.position[1]+1}}] = 0
  end
end

function GridWorldVis:start()
  super.start(self)
  
  self.screen:zero()
  self:drawPixel(true)

  return torch.Tensor(self.position)
end

function GridWorldVis:step(action)
  self:drawPixel(false)
  
  local reward, __, terminal = super.step(self, action)
  
  self:drawPixel(true)

  return reward, torch.Tensor(self.position), terminal
end

return GridWorldVis
