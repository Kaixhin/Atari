function optim.sharedRmsProp(opfunc, x, config, state)
  -- Get state
  local config = config or {}
  local state = state or config
  local lr = config.learningRate or 1e-2
  local momentum = config.momentum or 0.95
  local epsilon = config.rmsEpsilon or 0.01

  -- Evaluate f(x) and df/dx
  local fx, dfdx = opfunc(x)

  -- Initialise storage
  if not state.g then
    state.g = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
  end

  if not state.tmp then
    state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx)
  end

  state.g:mul(momentum):addcmul(1 - momentum, dfdx, dfdx)
  state.tmp:copy(state.g):add(epsilon):sqrt()

  -- Update x = x - lr x df/dx / tmp
  x:addcdiv(-lr, dfdx, state.tmp)

  -- Return x*, f(x) before optimisation
  return x, {fx}
end