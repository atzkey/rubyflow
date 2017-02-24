require 'node'

class Linear < Node
  def initialize(inputs, weights, bias)
    super([inputs, weights, bias])
  end

  # $ output = \sum_{i} x_{i} w_{i} + b $
  def forward
    _X, _W, _b = self.inbound_nodes.map(&:value)

   _X_W = _X * _W
    biases_broadcast = Matrix.build(_X_W.row_count, _X_W.column_count) do |row, column|
      _b[0, column]
    end

    self.value = _X_W + biases_broadcast
  end
end
