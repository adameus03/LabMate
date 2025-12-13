import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.example_libraries import optimizers
import numpy as np
from typing import Callable, List, Tuple, Dict
from functools import partial

class DenseNeuralNetwork:
    """
    Dense neural network with customizable loss function via callback.
    Architecture: 120 -> 168 -> 96 -> 66
    """
    
    def __init__(self, layer_sizes: List[int] = [120, 168, 96, 66], 
                 activation: str = 'relu', seed: int = 42):
        """
        Initialize the neural network.
        
        Args:
            layer_sizes: List of layer sizes (default: [120, 168, 96, 66])
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            seed: Random seed for initialization
        """
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.rng = jax.random.PRNGKey(seed)
        self.params = self._initialize_params()
        
    def _initialize_params(self) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Initialize network parameters using Xavier initialization."""
        params = []
        for i in range(len(self.layer_sizes) - 1):
            key, self.rng = jax.random.split(self.rng)
            in_size, out_size = self.layer_sizes[i], self.layer_sizes[i + 1]
            
            # Xavier initialization
            limit = jnp.sqrt(6.0 / (in_size + out_size))
            w = jax.random.uniform(key, (in_size, out_size), 
                                  minval=-limit, maxval=limit)
            b = jnp.zeros(out_size)
            params.append((w, b))
        return params
    
    def _activation(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply activation function."""
        if self.activation_name == 'relu':
            return jax.nn.relu(x)
        elif self.activation_name == 'tanh':
            return jnp.tanh(x)
        elif self.activation_name == 'sigmoid':
            return jax.nn.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation_name}")
    
    @partial(jit, static_argnums=(0,))
    def forward(self, params: List[Tuple[jnp.ndarray, jnp.ndarray]], 
                x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the network.
        
        Args:
            params: Network parameters
            x: Input data
            
        Returns:
            Network output in range [-1, 1]
        """
        for i, (w, b) in enumerate(params[:-1]):
            x = self._activation(jnp.dot(x, w) + b)
        
        # Last layer with tanh activation for [-1, 1] output range
        w, b = params[-1]
        return jnp.tanh(jnp.dot(x, w) + b)
    
    def train(self, 
              loss_callback: Callable[[], float],
              epochs: int = 100,
              learning_rate: float = 0.001,
              device: str = 'cpu',
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the network using Adam optimizer with custom loss function.
        
        Args:
            loss_callback: Function() -> scalar loss
                          This function takes no arguments. Access network params
                          via self.params and call self.forward(self.params, data)
                          to evaluate the network on any data you want.
            epochs: Number of training epochs
            learning_rate: Learning rate for Adam optimizer
            device: 'cpu', 'gpu', or 'tpu' - automatically handles device placement
            verbose: Print training progress
            
        Returns:
            Dictionary containing training history
        """
        # Move params to device
        self.params = [
            (self._to_device(w, device), self._to_device(b, device))
            for w, b in self.params
        ]
        
        # Initialize Adam optimizer
        opt_init, opt_update, get_params = optimizers.adam(learning_rate)
        opt_state = opt_init(self.params)
        
        # Create loss function that uses the callback
        def loss_fn(params):
            # Update self.params so callback can access current weights
            old_params = self.params
            self.params = params
            loss = loss_callback()
            self.params = old_params
            return loss
        
        # Gradient function
        grad_fn = grad(loss_fn)
        
        # Training step
        def step(i, opt_state):
            params = get_params(opt_state)
            g = grad_fn(params)
            return opt_update(i, g, opt_state)
        
        # Training loop
        history = {'loss': []}
        
        for epoch in range(epochs):
            # Update parameters
            opt_state = step(epoch, opt_state)
            
            # Calculate loss for monitoring
            self.params = get_params(opt_state)
            epoch_loss = float(loss_callback())
            history['loss'].append(epoch_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        # Update network parameters
        self.params = get_params(opt_state)
        return history
    
    def predict(self, x: jnp.ndarray, device: str = 'cpu') -> jnp.ndarray:
        """
        Make predictions.
        
        Args:
            x: Input data
            device: 'cpu', 'gpu', or 'tpu'
            
        Returns:
            Network predictions
        """
        x = self._to_device(x, device)
        return self.forward(self.params, x)
    
    def _to_device(self, data: jnp.ndarray, device: str) -> jnp.ndarray:
        """Move data to specified device."""
        if device == 'cpu':
            return jax.device_put(data, jax.devices('cpu')[0])
        elif device == 'gpu':
            return jax.device_put(data, jax.devices('gpu')[0])
        elif device == 'tpu':
            return jax.device_put(data, jax.devices('tpu')[0])
        else:
            return data