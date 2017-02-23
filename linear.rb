require 'node'

class Linear < Node
  def initialize(inputs, weights, bias)
    super([inputs, weights, bias])
  end

  # $ output = \sum_{i} x_{i} y_{i} + b $
  def forward
    inputs, weights, bias = self.inbound_nodes.map(&:value)
    self.value = inputs.zip(weights).reduce(0) { |r, (a, b)| r + a * b } + bias
  end
end
