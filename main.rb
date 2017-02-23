require 'rubyflow'
require 'input'
require 'node'
require 'add'

x, y = Input.new, Input.new
f = Add.new(x, y)

feed_dict = {x => 40, y => 2}

sorted_nodes = topological_sort(feed_dict)

puts forward_pass(f, sorted_nodes)
