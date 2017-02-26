require 'pry'
require 'matrix'

require 'rubyflow'

require 'test/unit/assertions'
include Test::Unit::Assertions

def assert_almost_equal(x, y, epsilon = 1e-5)
  assert (x - y).all? { |x| x.abs < epsilon }
end


def test_basic
  w, x, y, z = Input.new, Input.new, Input.new, Input.new
  f = Mul.new(Add.new(w, x, y), x, z)

  feed_dict = {w => 1, x => 2, y => 3, z => 4}

  graph = topological_sort(feed_dict)
  forward_pass(graph)

  assert_equal(48, f.value)
end

def test_linear
  _X, _W, _b = Input.new, Input.new, Input.new

  f = Linear.new(_X, _W, _b)

  xm = Matrix[[-1.0, -2.0]]
  wm = Matrix[[2.0, -3.0], [2.0, -3.0]]
  bv = Matrix[[-3.0, -5.0]]

  feed_dict = {_X => xm, _W => wm, _b => bv}

  graph = topological_sort(feed_dict)
  forward_pass(graph)

  assert_equal(Matrix[[-9.0, 4.0]], f.value)
end

def test_sigmoid
  _X, _W, _b = Input.new, Input.new, Input.new

  f = Linear.new(_X, _W, _b)
  g = Sigmoid.new(f)

  xm = Matrix[[-1.0, -2.0]]
  wm = Matrix[[2.0, -3.0], [2.0, -3.0]]
  bv = Matrix[[-3.0, -5.0]]

  feed_dict = {_X => xm, _W => wm, _b => bv}

  graph = topological_sort(feed_dict)
  forward_pass(graph)

  assert_almost_equal(Matrix[
    [1.234e-4, 9.820138e-1]
  ], g.value)
end

def test_mse
  y, a = Input.new, Input.new
  cost = MSE.new(y, a)

  y_ = Matrix[[1, 2, 3]]
  a_ = Matrix[[4.5, 5, 10]]

  feed_dict = {y => y_, a => a_}

  graph = topological_sort(feed_dict)

  forward_pass(graph)

  assert_equal(23.416667, cost.value.round(6))
end


def test_backpropagation
  _X, _W, _b = Input.new, Input.new, Input.new
  _y = Input.new

  f = Linear.new(_X, _W, _b)
  a = Sigmoid.new(f)
  cost = MSE.new(_y, a)

  xm = Matrix[[-1.0, -2.0], [-1.0, -2.0]]
  wm = Matrix[[2.0], [3.0]]
  bv = Matrix[[-3.0], [-3.0]]
  yv = Matrix[[1, 2]]

  feed_dict = {_X => xm, _W => wm, _b => bv, _y => yv}

  graph = topological_sort(feed_dict)
  forward_and_backward_pass(graph)

  assert_almost_equal(Matrix[
    [ -3.34017280e-05,  -5.01025919e-05],
    [ -6.68040138e-05,  -1.00206021e-04]],
    _X.gradients[_X])
end

test_basic
test_linear
test_sigmoid
test_mse
test_backpropagation
