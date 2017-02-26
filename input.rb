require 'node'

class Input < Node
  def initialize
    super
  end

  def forward(value = nil)
    self.value = value if value
  end

  def backward
    @gradients[self] = @outbound_nodes.map { |n| n.gradients[self] }.reduce(:+)
  end
end
