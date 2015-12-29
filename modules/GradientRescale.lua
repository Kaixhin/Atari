local GradientRescale, parent = torch.class('nn.GradientRescale', 'nn.Module')

function GradientRescale:__init(scaleFactor, inplace)
  parent.__init(self)
  self.scaleFactor = scaleFactor
  self.inplace = inplace
end

function GradientRescale:updateOutput(input)
  self.output = input
  return self.output
end

function GradientRescale:updateGradInput(input, gradOutput)
  if self.inplace then
    self.gradInput = gradOutput
  else
    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)
  end
  self.gradInput:mul(self.scaleFactor)
  return self.gradInput
end
