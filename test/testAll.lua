require 'torch' -- on travis luajit is invoked and this is needed

tester = torch.Tester()

tester:add(require 'test/testBinaryHeap')
tester:add(require 'test/testExperience')

tester:run()