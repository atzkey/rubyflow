require 'rubyflow'
require 'input'
require 'node'
require 'add'
require 'mul'

w, x, y, z = Input.new, Input.new, Input.new, Input.new
f = Mul.new(Add.new(w, x, y), x, z)

feed_dict = {w => 1, x => 2, y => 3, z => 4}

sorted_nodes = topological_sort(feed_dict)

puts forward_pass(f, sorted_nodes)
