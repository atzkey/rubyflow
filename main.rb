require 'rubyflow'

require 'test/unit/assertions'
include Test::Unit::Assertions

def test_basic
  w, x, y, z = Input.new, Input.new, Input.new, Input.new
  f = Mul.new(Add.new(w, x, y), x, z)

  feed_dict = {w => 1, x => 2, y => 3, z => 4}

  graph = topological_sort(feed_dict)

  assert(48 == forward_pass(f, graph))
end

def test_linear
  inputs, weights, bias = Input.new, Input.new, Input.new

  f = Linear.new(inputs, weights, bias)

  feed_dict = {
    inputs => [6, 14, 3],
    weights => [0.5, 0.25, 1.4],
    bias => 2
  }

  graph = topological_sort(feed_dict)

  assert(12.7 == forward_pass(f, graph))
end

test_basic
test_linear
