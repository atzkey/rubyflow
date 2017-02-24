# Mean Squared Error, loss evaluation node

class MSE < Node
  def initialize(y, a)
    super([y, a])
  end

  def forward
    # Reshape inputs from matrices to arrays
    ys = self.inbound_nodes[0].value.each(:all)
    as = self.inbound_nodes[1].value.each(:all)

    self.value = ys.zip(as).map { |y, a| (y - a)**2 }.mean
  end
end
