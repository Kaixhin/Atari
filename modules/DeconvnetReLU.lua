local DeconvnetReLU, parent = torch.class('nn.DeconvnetReLU', 'nn.ReLU')

function DeconvnetReLU:__init(p)
  parent.__init(self, p)
  self.deconv = false
end

function DeconvnetReLU:updateOutput(input)
  return parent.updateOutput(self, input)
end

function DeconvnetReLU:updateGradInput(input, gradOutput)
  if self.deconv then
    -- Backpropagate all positive error signals (irrelevant of positive inputs)
    if self.inplace then
      self.gradInput = gradOutput
    else
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
    end
    
    self.gradInput:cmul(torch.gt(gradOutput, 0):typeAs(gradOutput))
  else
    parent.updateGradInput(self, input, gradOutput)
  end

  return self.gradInput
end

function DeconvnetReLU:salientBackprop()
  self.deconv = true
end

function DeconvnetReLU:normalBackprop()
  self.deconv = false
end
