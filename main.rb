require 'matrix'

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
  _X, _W, _b = Input.new, Input.new, Input.new

  f = Linear.new(_X, _W, _b)

  xm = Matrix[[-1.0, -2.0], [-1, -2]]
  wm = Matrix[[2.0, -3], [2.0, -3]]
  bv = Matrix[[-3.0, -5]]

  feed_dict = {_X => xm, _W => wm, _b => bv}

  graph = topological_sort(feed_dict)

  assert(Matrix[[-9.0, 4.0], [-9.0, 4]] == forward_pass(f, graph))
end

def test_sigmoid
  _X, _W, _b = Input.new, Input.new, Input.new

  f = Linear.new(_X, _W, _b)
  g = Sigmoid.new(f)

  xm = Matrix[[-1.0, -2.0], [-1.0, -2.0]]
  wm = Matrix[[2.0, -3.0], [2.0, -3.0]]
  bv = Matrix[[-3.0, -5.0]]

  feed_dict = {_X => xm, _W => wm, _b => bv}

  graph = topological_sort(feed_dict)

  assert_equal(Matrix[
    [1.234e-4, 9.820138e-1],
    [1.234e-4, 9.820138e-1]], forward_pass(g, graph).map {|x| x.round(7)})
end

test_basic
test_linear
test_sigmoid
