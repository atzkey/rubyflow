require 'set'
require 'yaml'

require 'input'
require 'node'
require 'add'
require 'mul'
require 'linear'
require 'sigmoid'
require 'mse'

class Set
  def pop
    x = self.first
    self.delete(x)
    x
  end
end

class Array
  def mean
    reduce(&:+) / size
  end
end

# Sort generic nodes in topological order using Kahn's Algorithm.
#
#     `feed`: A dictionary where the key is an `Input` node and the value is the respective value feed to that node.
#
#     Returns a list of sorted nodes.
def topological_sort(feed)
  input_nodes = feed.keys.clone

  g = {}
  nodes = input_nodes.clone
  until nodes.empty?
    n = nodes.pop

    unless g.has_key?(n)
      g[n] = {in: Set.new, out: Set.new}
    end

    n.outbound_nodes.each do |m|
      unless g.has_key?(m)
        g[m] = {in: Set.new, out: Set.new}
      end

      g[n][:out] << m
      g[m][:in] << n
      nodes << m
    end
  end

  l = []
  s = Set.new(input_nodes)

  until s.empty?
    n = s.pop()

    if n.is_a?(Input)
      n.value = feed[n]
    end

    l << n
    n.outbound_nodes.each do |m|
      g[n][:out].delete(m)
      g[m][:in].delete(n)

      # if no other incoming edges add to S
      if g[m][:in].empty?
        s << m
      end
    end
  end

  l
end

# Performs a forward pass through a list of sorted nodes.
#
# Arguments:
#
#        `sorted_nodes`: A topologically sorted list of nodes.
#
def forward_pass(sorted_nodes)
    sorted_nodes.each(&:forward)
end

def backward_pass(sorted_nodes)
  sorted_nodes.reverse.each(&:backward)
end

def forward_and_backward_pass(sorted_nodes)
  forward_pass(sorted_nodes)
  backward_pass(sorted_nodes)
end
