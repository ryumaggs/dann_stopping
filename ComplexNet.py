import torch
import torch.nn as nn
import numpy as np


class ComplexLayer(nn.Module):
    """
    A complex-valued neural network layer where weights are represented by
    real and imaginary (phase) components.
    
    The forward pass computes two operations and averages them:
    1. Magnitude of complex weights multiplied by input
    2. Magnitude of (complex weights multiplied by input)
    """
    
    def __init__(self, in_features, out_features, bias=True):
        """
        Args:
            in_features: Size of input dimension
            out_features: Size of output dimension
            bias: If True, adds a learnable bias (default: True)
        """
        super(ComplexLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Real component of complex weights
        self.weight_real = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Imaginary component of complex weights
        self.weight_imag = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.weight_real)
        nn.init.xavier_uniform_(self.weight_imag)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def get_complex_weights(self):
        """Returns complex weights as a complex tensor"""
        return torch.complex(self.weight_real, self.weight_imag)
    
    def forward(self, x, printt: bool = False):
        """
        Forward pass computing the average of two operations:
        1) |W| @ x
        2) |W @ x|
        Then applies per-layer gain control to prevent explosion.

        Args:
            x: (batch_size, in_features)

        Returns:
            (batch_size, out_features)
        """
        # --- config defaults (safe even if you forget to define in __init__) ---
        eps = getattr(self, "eps", 1e-6)
        target_gain = 0.9 #getattr(self, "target_gain", 0.9)  # <= 1.0 recommended for stability
        use_softabs = False #getattr(self, "use_softabs", False)
        softabs_eps = 1e-6 #getattr(self, "softabs_eps", 1e-6)

        # Get complex weights: shape (out_features, in_features), dtype complex
        W_complex = self.get_complex_weights()

        # ----- Compute |W| (entrywise magnitude) -----
        # Use explicit formulation to be clear and stable for complex
        # absW: (out_features, in_features), real >= 0
        absW = torch.sqrt(W_complex.real.pow(2) + W_complex.imag.pow(2) + softabs_eps**2) if use_softabs else torch.abs(W_complex)

        # ----- Gain control (key stabilization) -----
        # Use a fast, safe upper bound on operator norm: ||A||_inf = max row sum
        # For A=|W|, this bounds amplification in infinity norm and is a strong practical brake.
        row_sum = absW.sum(dim=1)                    # (out_features,)
        gain_bound = row_sum.max().clamp_min(eps)    # scalar

        # Compute a multiplicative scale so typical per-layer amplification <= target_gain
        scale = (target_gain / gain_bound).to(x.dtype)

        if printt:
            # a few useful diagnostics
            # (avoid printing tensors huge; just scalars)
            print(f"[ComplexLayer] gain_bound(max row-sum |W|): {gain_bound.item():.4f}, scale: {scale.item():.6f}")

        # ----- Operation 1: |W| @ x -----
        # x: (B, in), absW.T: (in, out) -> (B, out)
        output_1 = torch.matmul(x, absW.t())

        # ----- Operation 2: |W @ x| -----
        # Convert x to complex once
        x_complex = torch.complex(x, torch.zeros_like(x))

        complex_result = torch.matmul(x_complex, W_complex.t())  # (B, out), complex

        # Magnitude of complex_result (optionally soft)
        if use_softabs:
            # soft magnitude for complex: sqrt(re^2 + im^2 + eps^2)
            output_2 = torch.sqrt(complex_result.real.pow(2) + complex_result.imag.pow(2) + softabs_eps**2)
        else:
            output_2 = torch.abs(complex_result)

        # ----- Average and apply gain control -----
        output = 0.5 * (output_1 + output_2)
        output = output * scale  # <-- critical: prevents layer-to-layer blow-up

        # Add bias if present (bias is not part of the multiplicative blow-up mechanism)
        if self.bias is not None:
            output = output + self.bias

        return output

    
    def extra_repr(self):
        """String representation for the layer"""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class ComplexNeuralNetwork(nn.Module):
    """
    A multi-layer complex-valued neural network.
    """
    
    def __init__(self, layer_sizes, activation=nn.ReLU(), use_bias=True):
        """
        Args:
            layer_sizes: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            activation: Activation function to use between layers (default: ReLU)
            use_bias: Whether to use bias in layers (default: True)
        """
        super(ComplexNeuralNetwork, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.activation = activation
        
        # Create layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(
                ComplexLayer(layer_sizes[i], layer_sizes[i + 1], bias=use_bias)
            )
    
    def forward(self, x):
        """
        Forward pass through all layers.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation to all layers except the last
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x
    
    def get_num_parameters(self):
        """Returns total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    # Create a network with architecture: 10 -> 20 -> 15 -> 5
    network = ComplexNeuralNetwork(
        layer_sizes=[10, 20, 15, 5],
        activation=nn.ReLU(),
        use_bias=True
    )
    
    print("Network architecture:")
    print(network)
    print(f"\nTotal parameters: {network.get_num_parameters()}")
    
    # Test forward pass
    batch_size = 32
    input_dim = 10
    x = torch.randn(batch_size, input_dim)
    
    output = network(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")