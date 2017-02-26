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
end
