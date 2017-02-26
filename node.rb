class Node
  attr_accessor :inbound_nodes, :outbound_nodes, :value, :gradients

  def initialize(inbound_nodes = [])
    @inbound_nodes = inbound_nodes
    @outbound_nodes = []
    @value = nil
    @gradients = {}

    inbound_nodes.each do |n|
      n.outbound_nodes << self
    end
  end

  def forward
    raise NotImplementedError
  end

  def backward
    raise NotImplementedError
  end
end
