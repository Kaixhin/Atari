local HuberCriterion, parent = torch.class('nn.HuberCriterion', 'nn.Criterion')

function HuberCriterion:__init(delta)
  parent.__init(self)
  self.delta = delta or 1 -- Boundary
  self.alpha = torch.Tensor() -- Residual
  self.sqMask = torch.ByteTensor()
  self.absMask = torch.ByteTensor()
end

function HuberCriterion:updateOutput(input, target)
  -- Calculate residual
  self.alpha = target - input

  -- Calculate masks
  self.sqMask = torch.abs(self.alpha):le(self.delta)
  self.absMask = torch.abs(self.alpha):gt(self.delta)

  -- Add squared loss
  self.output = torch.sum(torch.pow(self.alpha[self.sqMask], 2):mul(0.5))
  -- Add absolute loss
  self.output = self.output + torch.sum(torch.mul(self.alpha[self.absMask], self.delta):add(-0.5 * math.pow(self.delta, 2)))

  -- Average
  self.output = self.output / target:size(1)

  return self.output
end

function HuberCriterion:updateGradInput(input, target)
  self.gradInput:resizeAs(target)

  -- Calculate squared loss derivative
  self.gradInput[self.sqMask] = self.alpha[self.sqMask]
  -- Calculate absolute loss derivative
  self.gradInput[self.absMask] = torch.sign(self.alpha[self.absMask]):mul(self.delta)

  return self.gradInput
end

return nn.HuberCriterion
