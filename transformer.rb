require 'matrix'

class Transformer
	N = 6
	D = 512
	D_FF = 2048

	@embedding	# used as both input, and output pre-softmax linear transformation

	def initialize(vocab_size)
		@embedding = Matrix.zero(vocab_size, D)
	end

	def attention
	end

	def multihead_attention
	end

	def forward(batch)
		rows = []
		batch.each { |i|
			rows << @embedding[i, nil..nil]
		}	
		p Matrix[*rows]
	end
end

class Transformer
end

model = Transformer.new 100
model.forward([1, 2, 3])
