from dnn import DenseNeuralNetwork

oscillator_period = 120
#nn = DenseNeuralNetwork([120, 168, 96, 66])
nn = DenseNeuralNetwork([130, 182, 104, 71])

def loss_cb():
  # Insert custom input data
  # my_inputs = jnp.array([[1.0, 2.0, ..., 120_values], 
  #                        [3.0, 4.0, ..., 120_values]])  # shape: (batch_size, 120)
  
  # Feed forward through the network
  # outputs = nn.forward(nn.params, my_inputs)
    
  # outputs now has shape (batch_size, 66) with values in [-1, 1]
  
  # Calculate and return the loss
  # loss = ...  # loss calculation
  return float(1)

history = nn.train(loss_callback=loss_cb, epochs=100, device='cpu')
