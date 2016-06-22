local classic = require 'classic'
local qt = pcall(require, 'qt')
local image = require 'image'

local PseudoCount = require 'pseudocount'

local ExplorationBonus = classic.class('ExplorationBonus')


function ExplorationBonus:_init(opt)
  self.beta = opt.pseudoBeta
  self.zoom = 5
  self.window = qt and image.display({image=torch.FloatTensor(1,42,42), zoom=self.zoom})
  self.count = PseudoCount(42)
  self.counter = 0
  self.started = 0
  self.histLen = opt.histLen
  classic.strict(self)
end

function ExplorationBonus:bonus(screen)
  screen = image.scale(screen[self.histLen], 42, 42):mul(255):byte()

  local pseudoCount = self.count:pseudoCount(screen)
  local bonus = self.beta * math.pow(pseudoCount + 0.01, -.5)

  -- print(pseudoCount ..' : '.. bonus)
  return bonus
end



return ExplorationBonus
