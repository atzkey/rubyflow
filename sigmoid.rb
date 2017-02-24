require 'node'

class Sigmoid < Node
  def initialize(node)
    super([node])
  end

  def sigmoid(xs)
    xs.map { |x| 1.0 / (1 + Math.exp(-x)) }
  end
  private :sigmoid

  def forward
    self.value = sigmoid(self.inbound_nodes[0].value)
  end
end
