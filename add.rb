require 'node'

class Add < Node
  def initialize(x, y)
    super([x, y])
  end

  def forward
    self.value = self.inbound_nodes.map(&:value).sum
  end
end
