local _ = require 'moses'
local nn = require 'nn'
require 'cunn'
local cudnn = require 'cudnn'
local image = require 'image'

local model = {}

-- Processes the full screen for DQN input
local preprocess = function(observation)
  local input = torch.CudaTensor(observation:size(1), 1, 84, 84)

  -- Loop over received frames
  for f = 1, observation:size(1) do
    -- Convert to grayscale
    local frame = image.rgb2y(observation:select(1, f):float()) -- image does not work with CudaTensor
    -- Reduce 210x160 screen to 84x84
    input[{{f}, {}, {}, {}}] = image.scale(frame, 84, 84)
  end

  return input
end

-- Creates a DQN model
local createModel = function(A)
  local net = nn.Sequential()
  -- TODO: Work out how to get 4 observations
  net:add(cudnn.SpatialConvolution(1, 32, 8, 8, 4, 4))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(32, 64, 4, 4, 2, 2))
  net:add(cudnn.ReLU(true))
  net:add(cudnn.SpatialConvolution(64, 64, 3, 3, 1, 1))
  net:add(cudnn.ReLU(true))
  net:add(nn.View(64*7*7))
  net:add(nn.Linear(64*7*7, 512))
  net:add(cudnn.ReLU(true))
  net:add(nn.Linear(512, _.size(A)))
  net:cuda()

  return net
end

-- An agent must observe, act, and learn

-- Creates an agent from a DQN
model.createAgent = function(gameEnv, opt)
  local agent = {}
  local A = gameEnv:getActions()
  -- DQN
  local net = createModel(A)
  -- Model parameters θ
  local theta, dTheta = net:getParameters()
  -- Experience replay memory
  local experience = {}
  local expIndex = 1
  local t = 1 -- TODO: Connect with steps
  -- SARSA
  local s0 = nil
  local a0 = nil
  local r0 = nil
  local s1 = nil
  local a1 = nil
  
  --[[
  -- Create transition table.
  ---- assuming the transition table always gets floating point input
  ---- (Foat or Cuda tensors) and always returns one of the two, as required
  ---- internally it always uses ByteTensors for states, scaling and
  ---- converting accordingly
  local transition_args = {
    stateDim = self.state_dim, numActions = self.n_actions,
    histLen = self.hist_len, gpu = self.gpu,
    maxSize = self.replay_memory, histType = self.histType,
    histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
    bufferSize = self.bufferSize
  }

  self.transitions = dqn.TransitionTable(transition_args)
  --]]

  local learnFromTuple = function(_s0, _a0, _r0, _s1, _a1)
    -- Calculate max Q-value from next step
    local QMax = torch.max(net:forward(_s1), 1)
    -- Calculate target Y
    local Y = _r0 + opt.gamma*QMax
    -- Calculate error
    local err = torch.mul(theta, (Y - output[a]))
    -- Compute target Q value
    --[[
    //compute the target Q value
    var tmat = this.forwardQ(this.net, s1, false);
    var qmax = r0 + this.gamma * tmat.w[R.maxi(tmat.w)];

    // now predict
    var pred = this.forwardQ(this.net, s0, true);

    var tderror = pred.w[a0] - qmax;
    var clamp = this.tderror_clamp;
    if(Math.abs(tderror) > clamp) {  // huber loss to robustify
      if(tderror > clamp) tderror = clamp;
      if(tderror < -clamp) tderror = -clamp;
    }
    pred.dw[a0] = tderror;
    this.lastG.backward(); // compute gradients on net params

    // update net
    R.updateNet(this.net, this.alpha);
    return tderror;
    --]]
    
    return 0
  end

  -- Performs an action on the environment
  agent.act = function(self, observation, trainMode)
    local aIndex
    -- Choose action by ε-greedy
    if math.random() < opt.epsilon then 
      aIndex = torch.random(1, _.size(A))
    else
      local state = preprocess(observation)
      local __, ind = torch.max(net:forward(state), 1)
      aIndex = ind[1]
    end

    -- SARSA update
    s0 = s1
    a0 = a1
    s1 = state
    a1 = aIndex

    return aIndex
  end

  -- Perform a learning step from the latest reward
  agent.learn = function(self, reward)
    if s0 ~= nil and opt.alpha > 0 then
      print(s0:size())
      print(a0)
      print(r0)
      print(s1:size())
      print(a1)
      -- Calculate TD error
      learnFromTuple(s0, a0, r0, s1, a1)

      -- Decide whether to store in replay memory
      if t % memSampleFreq == 0 then
        experience[expIndex] = {s0, a0, r0, s1, a1}
        expIndex = expIndex + 1
        -- Roll back to beginning if overflow
        if expIndex > opt.expReplMem then
          expIndex = 1
        end
      end
      t = t + 1

      -- Sample and learn from experience in replay memory
      for r = 1, 10 do -- TODO: Learning steps per iteration - minibatch size?
        local randomIndex = torch.random(1, _.size(experience))
        local replay = experience[randomIndex]
        learnFromTuple(replay[1], replay[2], replay[3], replay[4], replay[5])
      end
    end

    r0 = reward
  end

  return agent
end

return model
