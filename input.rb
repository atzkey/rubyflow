require 'node'

class Input < Node
  def initialize
    super
  end

  def forward(value = nil)
    self.value = value if value
  end
end
