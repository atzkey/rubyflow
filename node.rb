class Node
  attr_accessor :inbound_nodes, :outbound_nodes, :value

  def initialize(inbound_nodes = [])
    self.inbound_nodes = inbound_nodes
    self.outbound_nodes = []
    self.value = nil

    self.inbound_nodes.each do |n|
      n.outbound_nodes << self
    end
  end

  def forward()
    raise NotImplementedError
  end
end
