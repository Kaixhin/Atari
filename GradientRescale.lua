local GradientRescale, parent = torch.class('nn.GradientRescale', 'nn.Module')

function GradientRescale:__init(scaleFactor)
  parent.__init(self)
  self.scaleFactor = scaleFactor
end

function GradientRescale:updateOutput(input)
  self.output = input
  return self.output
end

function GradientRescale:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(gradOutput)
  self.gradInput:copy(gradOutput)
  self.gradInput:mul(self.scaleFactor)
  return self.gradInput
end
