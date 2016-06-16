-- Creates aggregator module for a dueling architecture based on a number of discrete actions
local DuelAggregator = function(m)
  local aggregator = nn.Sequential()
  local aggParallel = nn.ParallelTable()
  
  -- Advantage duplicator (for calculating and subtracting mean)
  local advDuplicator = nn.Sequential()
  local advConcat = nn.ConcatTable()
  advConcat:add(nn.Identity())
  -- Advantage mean duplicator
  local advMeanDuplicator = nn.Sequential()
  advMeanDuplicator:add(nn.Mean(1, 1))
  advMeanDuplicator:add(nn.Replicate(m, 2, 2))
  advConcat:add(advMeanDuplicator)
  advDuplicator:add(advConcat)
  -- Subtract mean from advantage values
  advDuplicator:add(nn.CSubTable())
  
  -- Add value and advantage duplicators
  aggParallel:add(nn.Replicate(m, 2, 2))
  aggParallel:add(advDuplicator)

  -- Calculate Q^ = V^ + A^
  aggregator:add(aggParallel)
  aggregator:add(nn.CAddTable())

  return aggregator
end

return DuelAggregator
