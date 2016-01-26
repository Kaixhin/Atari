local class = require 'classic'
require 'classic.torch' -- Enables serialisation

local Singleton = classic.class('Singleton')

function Singleton:_init(fields)
  -- Check for existing object
  if not Singleton.getInstance() then
    -- Populate new object with data
    for k, v in pairs(fields) do
      self[k] = v
    end

    -- Set static instance
    Singleton.static.instance = self
  end
end

-- Gets static instance
function Singleton.static.getInstance()
  return Singleton.static.instance
end

-- Sets static instance
function Singleton.static.setInstance(inst)
  Singleton.static.instance = inst
end

return Singleton
