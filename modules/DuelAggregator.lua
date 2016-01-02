-- Creates aggregator module for a dueling architecture based on a number of discrete actions
local DuelAggregator = function(m)
  local aggregator = nn.Sequential()
  local aggParallel = nn.ParallelTable()

  -- Value duplicator (for each action)
  local valDuplicator = nn.Sequential()
  local valConcat = nn.ConcatTable()
  for a = 1, m do
    valConcat:add(nn.Identity())
  end
  valDuplicator:add(valConcat)
  valDuplicator:add(nn.JoinTable(1, 1))

  -- Advantage duplicator (for calculating and subtracting mean)
  local advDuplicator = nn.Sequential()
  local advConcat = nn.ConcatTable()
  advConcat:add(nn.Identity())
  -- Advantage mean duplicator
  local advMeanDuplicator = nn.Sequential()
  advMeanDuplicator:add(nn.Mean(1, 1))
  local advMeanConcat = nn.ConcatTable()
  for a = 1, m do
    advMeanConcat:add(nn.Identity())
  end
  advMeanDuplicator:add(advMeanConcat)
  advMeanDuplicator:add(nn.JoinTable(1, 1))
  advConcat:add(advMeanDuplicator)
  advDuplicator:add(advConcat)
  -- Subtract mean from advantage values
  advDuplicator:add(nn.CSubTable())
  
  -- Add value and advantage duplicators
  aggParallel:add(valDuplicator)
  aggParallel:add(advDuplicator)

  -- Calculate Q^ = V^ + A^
  aggregator:add(aggParallel)
  aggregator:add(nn.CAddTable())

  return aggregator
end

return DuelAggregator
