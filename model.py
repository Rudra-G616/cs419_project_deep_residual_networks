import numpy as np

class ResNet:
    def __init__(self, input_dim, num_classes, num_blocks):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        
        # Initialize weights for ResNet
        self.weights = self.initialize_weights()
    
    def initialize_weights(self):
        weights = {}
        for i in range(self.num_blocks):
            weights[f'W{i+1}'] = np.random.randn(self.input_dim, self.input_dim) * 0.01
            weights[f'b{i+1}'] = np.zeros((1, self.input_dim))
        weights['W_out'] = np.random.randn(self.input_dim, self.num_classes) * 0.01
        weights['b_out'] = np.zeros((1, self.num_classes))
        return weights

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def forward_block(self, x, W, b):
        z = np.dot(x, W) + b
        out = self.relu(z)
        return out, z
    
    def residual_block(self, x, W, b):
        identity = x
        out, z = self.forward_block(x, W, b)
        out += identity  # Skip connection
        return out, z
    
    def forward(self, x):
        activations = []
        z_cache = []
        for i in range(self.num_blocks):
            W, b = self.weights[f'W{i+1}'], self.weights[f'b{i+1}']
            x, z = self.residual_block(x, W, b)
            activations.append(x)
            z_cache.append(z)
        W_out, b_out = self.weights['W_out'], self.weights['b_out']
        output = np.dot(x, W_out) + b_out
        return output, activations, z_cache
    
    def backward(self, x, y, output, activations, z_cache, learning_rate=0.01):
        m = y.shape[0]
        grads = {}
        
        # Output layer gradients
        dz = (output - y) / m
        grads['W_out'] = np.dot(activations[-1].T, dz)
        grads['b_out'] = np.sum(dz, axis=0, keepdims=True)
        
        # Backpropagate through residual blocks
        for i in reversed(range(self.num_blocks)):
            dz = np.dot(dz, self.weights['W_out'].T)
            dz = dz * self.relu_derivative(z_cache[i])
            grads[f'W{i+1}'] = np.dot(activations[i-1].T, dz) if i > 0 else np.dot(x.T, dz)
            grads[f'b{i+1}'] = np.sum(dz, axis=0, keepdims=True)
            dz = dz * (1 + self.relu_derivative(z_cache[i]))  # Residual contribution
        
        # Update weights
        for key in self.weights:
            self.weights[key] -= learning_rate * grads[key]
    
    def train(self, x, y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            output, activations, z_cache = self.forward(x)
            self.backward(x, y, output, activations, z_cache, learning_rate)
            if epoch % 10 == 0:
                loss = np.mean((output - y) ** 2)
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def predict(self, x):
        output, _, _ = self.forward(x)
        return np.argmax(output, axis=1)

# Example Usage:
if __name__ == "__main__":
    # Dummy data
    np.random.seed(42)
    x = np.random.randn(100, 64)  # 100 samples, 64 features
    y = np.eye(10)[np.random.choice(10, 100)]  # 10 classes
    
    model = ResNet(input_dim=64, num_classes=10, num_blocks=3)
    model.train(x, y, epochs=100, learning_rate=0.01)
    predictions = model.predict(x)
    print("Predictions:", predictions)
