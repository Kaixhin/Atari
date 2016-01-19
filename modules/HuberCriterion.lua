local HuberCriterion, parent = torch.class('nn.HuberCriterion', 'nn.Criterion')

function HuberCriterion:__init(delta)
  parent.__init(self)
  self.delta = delta or 1 -- Boundary
  self.alpha = torch.Tensor() -- Residual
end

function HuberCriterion:updateOutput(input, target)
  -- Calculate residual
  self.alpha = target - input

  self.absAlpha = torch.abs(self.alpha)
  self.diffAlpha = torch.cmin(self.absAlpha, self.delta)

  self.output = torch.cmul(self.diffAlpha, self.absAlpha:mul(2):add(-self.diffAlpha)):mul(0.5):mean()
  
  return self.output
end

function HuberCriterion:updateGradInput(input, target)
  self.gradInput:resizeAs(target)

  self.gradInput = self.alpha:sign():cmul(self.diffAlpha)

  return self.gradInput
end

return nn.HuberCriterion
