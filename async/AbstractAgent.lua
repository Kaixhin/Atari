local classic = require 'classic'

local AbstractAgent = classic.class('AbstractAgent')


AbstractAgent:mustHave('observe')
AbstractAgent:mustHave('training')
AbstractAgent:mustHave('evaluate')

return AbstractAgent