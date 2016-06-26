local MinDim, parent = torch.class('nn.MinDim', 'nn.Module')

local function _assertTensor(t)
   assert(torch.isTensor(t), "This module only works on tensor")
end

function MinDim:__init(pos, minInputDims)
   parent.__init(self)
   self.pos = pos or error('the position to insert singleton dim not specified')
   self:setMinInputDims(minInputDims)
end

function MinDim:setMinInputDims(numInputDims)
   self.numInputDims = numInputDims
   return self
end

function MinDim:updateOutput(input)
   _assertTensor(input)
   self.output = input
   if input:dim() < self.numInputDims then
     nn.utils.addSingletonDimension(self.output, input, self.pos)
   end
   return self.output
end

function MinDim:updateGradInput(input, gradOutput)
   _assertTensor(input)
   _assertTensor(gradOutput)
   assert(input:nElement() == gradOutput:nElement())
   self.gradInput:view(gradOutput, input:size())
   return self.gradInput
end
