local classic = require 'classic'
local qt = pcall(require, 'qt')
local image = require 'image'

local PseudoCount = require 'PseudoCount'

local ExplorationBonus = classic.class('ExplorationBonus')


function ExplorationBonus:_init(opt)
  self.beta = opt.pseudoBeta
  self.zoom = 5
  self.window = qt and image.display({image=torch.FloatTensor(1,42,42), zoom=self.zoom})
  self.count = PseudoCount(42)
  self.counter = 0
  self.started = 0
  classic.strict(self)
end

function ExplorationBonus:bonus(screen)
  screen = image.rgb2y(screen)
  screen = image.scale(screen, 42, 42)
  print(self.count:pseudoCount(screen))

  if self.started == 0 then self.started = torch.tic() end
  self.counter = self.counter + 1
  local since = torch.toc(self.started)
  local speed = self.counter / since
  print('speed='.. speed)

--  if true then return end

  -- TODO to Bytes
  self.window = image.display({image=screen, zoom=self.zoom, win=self.window})

end



return ExplorationBonus