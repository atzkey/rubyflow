require 'node'

class Linear < Node
  def initialize(inputs, weights, bias)
    super([inputs, weights, bias])
  end

  # $ output = \sum_{i} x_{i} w_{i} + b $
  def forward
    _X, _W, _b = @inbound_nodes.map(&:value)

    @value = _X * _W + _b
  end

  def backward
    inbound_nodes.each do |n|
      self.gradients[n] = Matrix.zero(n.value.row_count, n.value.column_count)
    end

    _X, _W, _b = self.inbound_nodes
    outbound_nodes.each do |n|
      grad_cost = n.gradients[self]
      self.gradients[_X] += grad_cost * _W.value.t
      self.gradients[_W] += _X.value.t * grad_cost
      self.gradients[_b] = self.gradients[_b].map { |e| e + grad_cost.each.sum }
    end
  end
end
