require 'node'

class Add < Node
  def initialize(*inputs)
    super(inputs)
  end

  def forward
    self.value = self.inbound_nodes.map(&:value).sum
  end
end
