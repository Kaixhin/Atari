local GuidedReLU, parent = torch.class('nn.GuidedReLU', 'nn.ReLU')

function GuidedReLU:__init(p)
  parent.__init(self, p)
  self.guide = false
end

function GuidedReLU:updateOutput(input)
  return parent.updateOutput(self, input)
end

function GuidedReLU:updateGradInput(input, gradOutput)
  parent.updateGradInput(self, input, gradOutput)
  if self.guide then
    -- Only backpropagate positive error signals
    self.gradInput:cmul(torch.gt(gradOutput, 0):typeAs(gradOutput))
  end
  return self.gradInput
end

function GuidedReLU:salientBackprop()
  self.guide = true
end

function GuidedReLU:normalBackprop()
  self.guide = false
end
