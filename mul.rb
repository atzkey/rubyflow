require 'node'

class Mul < Node
  def initialize(*inputs)
    super(inputs)
  end

  def forward
    self.value = self.inbound_nodes.map(&:value).reduce(&:*)
  end
end
