import tflearn

def isEven(num):
	data = [[1,0,0,0,0,0,0,0,0,0],
		[0,1,0,0,0,0,0,0,0,0],
		[0,0,1,0,0,0,0,0,0,0],
		[0,0,0,1,0,0,0,0,0,0],
		[0,0,0,0,1,0,0,0,0,0],
		[0,0,0,0,0,1,0,0,0,0],
		[0,0,0,0,0,0,1,0,0,0],
		[0,0,0,0,0,0,0,1,0,0],
		[0,0,0,0,0,0,0,0,1,0],
		[0,0,0,0,0,0,0,0,0,1]	]

	labels = [ 	[0, 1],
		   	[1, 0],
			[0, 1],
		   	[1, 0],
			[0, 1],
		   	[1, 0],
			[0, 1],
		   	[1, 0],
			[0, 1],
		   	[1, 0] ]

	# Build neural network
	net = tflearn.input_data(shape=[None, 10])
	net = tflearn.fully_connected(net, 5)
	net = tflearn.fully_connected(net, 2, activation='softmax')
	net = tflearn.regression(net)

	# Define model
	model = tflearn.DNN(net)
	# Start training (apply gradient descent algorithm)
	model.fit(data, labels, n_epoch=2000, batch_size=10, show_metric=True)

	# Predict even or odd
	pred = model.predict([data[num % 10]])

	return pred[0][1] > 0.5

print(isEven(3))

