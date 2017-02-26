# Mean Squared Error, loss evaluation node

class MSE < Node
  def initialize(y, a)
    super([y, a])
  end

  def forward
    # Reshape inputs from matrices to arrays
    ys = self.inbound_nodes[0].value.each(:all)
    as = self.inbound_nodes[1].value.each(:all)

    @diff = ys.zip(as).map { |y, a| y - a }
    @size = @diff.size

    @value = @diff.map { |x| x**2 }.mean
  end

  def backward
    @gradients[inbound_nodes.first] = Matrix[@diff.map { |x| (2 / @size) * x }]
    @gradients[inbound_nodes.last] = Matrix[@diff.map { |x| (-2 / @size) * x }]
  end
end
