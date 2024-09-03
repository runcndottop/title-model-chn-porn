require 'matrix'

class Transformer
	N = 6	# number of encoder blocks
	D = 512	# word embedding dimension
	D_FF = 2048	# feed forward dimension
	H = 8	# number of heads for multihead attention

	@embedding	# used as both input, and output pre-softmax linear transformation

	def initialize(vocab_size, seq_len)
		@embedding = Matrix.zero(vocab_size, D)
		@seq_len = seq_len
	end

	def attention
	end

	def multihead_attention
	end

	def softmax(m)	# m is a matrix
		# compute softmax for each entry along the row
		m.row_vectors.collect {|r|
			dtr = 0
			r.each {|e|
				dtr += Math.exp(e)
			}

			r.collect {|e|
				Math.exp(e)/dtr
			}
		}
	end

	def forward(batch)
		# look up embeddings
		rows = []
		batch.each { |i|
			rows << @embedding[i, nil..nil]
		}	
		embedded = Matrix[*rows].collect! { |e| e * Math.sqrt(D) }
		# add positional encoding
		embedded += Matrix.build(batch.count, D) {|i, j|
			# i%seq_len is the position in the sequence
			if j%2 == 1
				Math.cos((i%@seq_len)/10000**((j-1)/D))
			else
				Math.sin((i%@seq_len)/10000**(j/D))
			end
		}
		p softmax(Matrix[[1,2,3], [0,0,0]])
	end
end

class Transformer
end

model = Transformer.new 100, 2
model.forward([1, 2, 3, 4])
