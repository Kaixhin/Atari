local BinaryHeap = require 'structures/BinaryHeap'
local tds = require 'tds'

local Test = torch.TestSuite()
local standalone = tester == nil
if standalone then
  tester = torch.Tester()
end


function Test:BinaryHeap_Test()
  local heap = BinaryHeap(1000)
  local vec = tds.Vec()

  for i=1,100 do
    local r = torch.random(100)
    vec[#vec+1] = r
    heap:insert(r,r*2)
  end

  vec:sort(function(a,b) return a > b end)

  tester:eq(heap:findMax(), vec[1])

  for i=1,100 do
    local entry = heap:pop()
    local r = vec[i]

    tester:eq(entry[1], r)
    tester:eq(entry[2], r*2)
  end
end



if standalone then
  tester:add(Test)
  tester:run()
end

return Test
