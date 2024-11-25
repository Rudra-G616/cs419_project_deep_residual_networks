import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import random

# Seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Dropout Layer
def dropout(X, keep_prob):
    mask = np.random.rand(*X.shape) < keep_prob
    return X * mask / keep_prob, mask

# Weight Initialization with He initialization
def init_weights(shape, activation='relu'):
    if activation == 'relu':
        return np.random.randn(*shape) * np.sqrt(2.0 / shape[0])
    return np.random.randn(*shape) * 0.01

# Data Loading and Preprocessing
def load_mnist_data():
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    
    # Convert to NumPy and normalize
    X_train = train_dataset.data.numpy().reshape(-1, 1, 28, 28) / 255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, 1, 28, 28) / 255.0
    y_test = test_dataset.targets.numpy()

    unique_indices_train = random.sample(range(len(X_train)), 500)
    unique_indices_test = random.sample(range(len(X_test)), 200)
    X_train, y_train = X_train[unique_indices_train], y_train[unique_indices_train]
    X_test, y_test = X_test[unique_indices_test], y_test[unique_indices_test]
    
    return X_train, y_train, X_test, y_test

# Activation Functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Convolution and Pooling Functions
def conv2d(X, W, b, stride=1, pad=1):
    n, c, h, w = X.shape
    f, _, fh, fw = W.shape
    
    # Padding
    X_pad = np.pad(X, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
    
    # Output dimensions
    oh = (h + 2*pad - fh) // stride + 1
    ow = (w + 2*pad - fw) // stride + 1
    
    # Im2col for efficiency
    col_X = np.lib.stride_tricks.as_strided(
        X_pad, 
        shape=(n, c, fh, fw, oh, ow),
        strides=(X_pad.strides[0], X_pad.strides[1], 
                 X_pad.strides[2], X_pad.strides[3], 
                 X_pad.strides[2]*stride, X_pad.strides[3]*stride)
    ).reshape(-1, c*fh*fw)
    
    col_W = W.reshape(f, -1).T
    
    out = np.dot(col_X, col_W).T + b.reshape(-1, 1)
    out = out.T.reshape(n, f, oh, ow)
    
    return out

def max_pool2d(X, pool_size=2, stride=2):
    n, c, h, w = X.shape
    
    oh = (h - pool_size) // stride + 1
    ow = (w - pool_size) // stride + 1
    
    output = np.zeros((n, c, oh, ow))
    
    for i in range(oh):
        for j in range(ow):
            h_start = i * stride
            h_end = h_start + pool_size
            w_start = j * stride
            w_end = w_start + pool_size
            
            output[:, :, i, j] = np.max(X[:, :, h_start:h_end, w_start:w_end], axis=(2, 3))
    
    return output

# ResNet Block
def resnet_block(X, W1, b1, W2, b2, W3, b3):
    # Conv1
    conv1 = conv2d(X, W1, b1, stride=1, pad=1)
    relu1 = relu(conv1)
    
    # Conv2
    conv2 = conv2d(relu1, W2, b2, stride=1, pad=1)
    
    # Shortcut connection
    shortcut = X
    
    # Final output
    output = relu(conv2 + shortcut)
    
    return output

# Forward Propagation with Dropout
def forward_prop(X, params, is_training=True):
    # ResNet Layers
    X1 = resnet_block(X, 
                      params['W1'], params['b1'], 
                      params['W2'], params['b2'], 
                      params['W3'], params['b3'])
    
    pool1 = max_pool2d(X1)
    
    # Flatten
    flattened = pool1.reshape(pool1.shape[0], -1)
    
    # Dropout during training
    dropout_mask = None
    if is_training:
        flattened, dropout_mask = dropout(flattened, keep_prob=0.5)
    
    # Fully Connected Layer
    fc_out = np.dot(flattened, params['Wfc']) + params['bfc']
    
    # Softmax
    probs = softmax(fc_out)
    
    return probs, {
        'X': X, 'X1': X1, 'pool1': pool1, 
        'flattened': flattened, 'fc_out': fc_out,
        'dropout_mask': dropout_mask
    }

# Loss and Accuracy
def cross_entropy_loss(probs, y):
    m = y.shape[0]
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(m), y] = 1
    loss = -np.sum(one_hot * np.log(probs + 1e-8)) / m
    return loss

def compute_accuracy(probs, y):
    predictions = np.argmax(probs, axis=1)
    return np.mean(predictions == y)

# Adam Optimizer
def adam_init(params):
    adam_cache = {}
    for key in params.keys():
        adam_cache[f'{key}_m'] = np.zeros_like(params[key])
        adam_cache[f'{key}_v'] = np.zeros_like(params[key])
    return adam_cache

def adam_update(params, grads, cache, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    for key in params.keys():
        # Update biased first moment estimate
        cache[f'{key}_m'] = beta1 * cache[f'{key}_m'] + (1 - beta1) * grads[key]
        
        # Update biased second raw moment estimate
        cache[f'{key}_v'] = beta2 * cache[f'{key}_v'] + (1 - beta2) * (grads[key] ** 2)
        
        # Compute bias-corrected first moment estimate
        m_hat = cache[f'{key}_m'] / (1 - beta1)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = cache[f'{key}_v'] / (1 - beta2)
        
        # Update parameters
        params[key] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return params, cache

# Backpropagation with L2 Regularization
def backward_prop(params, cache, y, l2_lambda=0.001):
    m = y.shape[0]
    probs, _ = forward_prop(cache['X'], params, is_training=False)
    
    # Softmax gradient
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(m), y] = 1
    dfc_out = probs - one_hot
    
    # FC Layer Gradients with L2 Regularization
    dWfc = np.dot(cache['flattened'].T, dfc_out) / m + l2_lambda * params['Wfc']
    dbfc = np.sum(dfc_out, axis=0) / m
    
    # Dropout gradient
    if cache['dropout_mask'] is not None:
        dflattened = np.dot(dfc_out, params['Wfc'].T) * cache['dropout_mask'] / 0.5
    else:
        dflattened = np.dot(dfc_out, params['Wfc'].T)
    
    # Reshape
    dpool1 = dflattened.reshape(cache['pool1'].shape)
    
    # Gradient through ResNet Block with L2 Regularization
    dW2 = np.zeros_like(params['W2']) + l2_lambda * params['W2']
    db2 = np.zeros_like(params['b2'])
    
    # Simplified gradient for the block
    dX1 = dpool1  # Simplified for brevity
    
    grads = {
        'Wfc': dWfc, 
        'bfc': dbfc, 
        'W1': np.zeros_like(params['W1']) + l2_lambda * params['W1'], 
        'b1': np.zeros_like(params['b1']), 
        'W2': dW2, 
        'b2': db2, 
        'W3': np.zeros_like(params['W3']) + l2_lambda * params['W3'], 
        'b3': np.zeros_like(params['b3'])
    }
    
    return grads

# Training Function
def train_resnet(X_train, y_train, X_test, y_test):
    # Model Parameters with He initialization
    params = {
        'W1': init_weights((16, 1, 3, 3), 'relu'),
        'b1': init_weights((16,)),
        'W2': init_weights((16, 16, 3, 3), 'relu'),
        'b2': init_weights((16,)),
        'W3': init_weights((1, 16, 1, 1), 'relu'),
        'b3': init_weights((1,)),
        'Wfc': init_weights((16*14*14, 10)),
        'bfc': init_weights((10,))
    }
    
    # Training Parameters
    epochs = 50
    batch_size = 64
    early_stopping_patience = 5
    l2_lambda = 0.001  # L2 regularization strength
    
    # Tracking
    train_losses, val_losses = [], []
    
    # Adam Optimizer
    adam_cache = adam_init(params)
    
    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Batch training
        epoch_train_losses = []
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train_shuffled[i:i+batch_size]
            batch_y = y_train_shuffled[i:i+batch_size]
            
            # Forward Prop
            probs, cache = forward_prop(batch_X, params)
            
            # Compute Loss with L2 Regularization
            ce_loss = cross_entropy_loss(probs, batch_y)
            l2_loss = sum(np.sum(np.square(params[key])) for key in ['W1', 'W2', 'W3', 'Wfc']) * l2_lambda / 2
            loss = ce_loss + l2_loss
            
            epoch_train_losses.append(loss)
            
            # Backpropagation
            grads = backward_prop(params, cache, batch_y, l2_lambda)
            
            # Optimizer Update
            params, adam_cache = adam_update(params, grads, adam_cache)
        
        # Training Loss and Accuracy
        train_probs, _ = forward_prop(X_train, params, is_training=False)
        train_loss = np.mean(epoch_train_losses)
        train_acc = compute_accuracy(train_probs, y_train)
        train_losses.append(train_loss)
        
        # Validation Loss and Accuracy
        val_probs, _ = forward_prop(X_test, params, is_training=False)
        val_loss = cross_entropy_loss(val_probs, y_test)
        val_acc = compute_accuracy(val_probs, y_test)
        val_losses.append(val_loss)
        
        # Print Epoch Info
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
    
    return params, train_losses, val_losses

# Visualization Function
def visualize_predictions(X_test, y_test, params, num_samples=5):
    probs, _ = forward_prop(X_test, params, is_training=False)
    predictions = np.argmax(probs, axis=1)
    
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X_test[i].squeeze(), cmap='gray')
        plt.title(f"True: {y_test[i]}, Pred: {predictions[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualization for Losses
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Main Execution
def main():
    # Load Data
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    # Train Model
    params, train_losses, val_losses = train_resnet(X_train, y_train, X_test, y_test)
    
    # Visualize Losses
    plot_losses(train_losses, val_losses)
    
    # Visualize Predictions
    visualize_predictions(X_test, y_test, params)

# Run the main function
main()