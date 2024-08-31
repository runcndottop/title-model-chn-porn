require 'matrix'

class Transformer
	N = 6
	D = 512
	D_FF = 2048

	@embedding

	def initialize(vocab_size)
		@embedding = Matrix.build(vocab_size, D)
	end

	def attention
	end

	def multihead_attention
	end
end

class Transformer
end
