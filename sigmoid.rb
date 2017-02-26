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

  def backward
    inbound_nodes.each do |n|
      self.gradients[n] = Matrix.zero(n.value.row_count, n.value.column_count)
    end

    outbound_nodes.each do |n|
      grad_cost = n.gradients[self]
      sigmoid = self.value

      # sigmoid' * grad_cost == sigmoid * (1 - sigmoid) * grad_cost
      self.gradients[self.inbound_nodes.first] += Matrix[
        *sigmoid.map { |s| s * (1 - s) }.zip(grad_cost).map { |x, y| [x * y] }
      ]
    end
  end
end
