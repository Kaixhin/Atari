require 'torch' -- on travis luajit is invoked and this is needed

tester = torch.Tester()

require 'test/testBinaryHeap'

tester:run()