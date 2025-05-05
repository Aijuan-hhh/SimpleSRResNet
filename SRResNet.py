import numpy as np


class Layer:
    """Base layer class with required interface methods"""

    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, dout):
        raise NotImplementedError


def im2col(input_data, kernel_h, kernel_w, stride=1, pad=0):
    """Convert input data to column format for efficient convolution"""
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, kernel_h, kernel_w, out_h, out_w))

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, kernel_h, kernel_w, stride=1, pad=0):
    """Convert column data back to image format for backpropagation"""
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - kernel_h) // stride + 1
    out_w = (W + 2 * pad - kernel_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * pad, W + 2 * pad))

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    if pad > 0:
        img = img[:, :, pad:-pad, pad:-pad]

    return img


class Conv2D(Layer):
    """2D Convolutional layer"""

    def __init__(self, filters, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def initialize(self, input_shape):
        c, h, w = input_shape
        scale = np.sqrt(2.0 / (c * self.kernel_size * self.kernel_size))  # He initialization
        self.params["W"] = np.random.randn(self.filters, c, self.kernel_size, self.kernel_size).astype(np.float32) * scale
        self.params["b"] = np.zeros(self.filters,dtype=np.float32)

        h_out = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        return (self.filters, h_out, w_out)

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, c, h, w = inputs.shape

        h_out = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        w_out = (w + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.col = im2col(inputs, self.kernel_size, self.kernel_size, self.stride, self.padding)
        self.col_W = self.params["W"].reshape(self.filters, -1).T

        out = np.dot(self.col, self.col_W) + self.params["b"]
        out = out.reshape(batch_size, h_out, w_out, self.filters).transpose(0, 3, 1, 2)

        return out

    def backward(self, dout):
        batch_size, filters, h_out, w_out = dout.shape

        dout = dout.transpose(0, 2, 3, 1).reshape(-1, filters)
        self.grads["b"] = np.sum(dout, axis=0)
        self.grads["W"] = np.dot(self.col.T, dout)
        self.grads["W"] = self.grads["W"].transpose(1, 0).reshape(self.filters, -1, self.kernel_size, self.kernel_size)

        dcol = np.dot(dout, self.col_W.T)
        dinputs = col2im(dcol, self.inputs.shape, self.kernel_size, self.kernel_size,
                         self.stride, self.padding)

        return dinputs


class BatchNorm2D(Layer):
    """2D Batch Normalization layer"""

    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.running_mean = None
        self.running_var = None
        self.cache = None

    def initialize(self, input_shape):
        self.params["gamma"] = np.ones(self.num_features)
        self.params["beta"] = np.zeros(self.num_features)
        self.running_mean = np.zeros(self.num_features)
        self.running_var = np.ones(self.num_features)
        return input_shape

    def forward(self, inputs, training=True):
        batch_size, c, h, w = inputs.shape
        x_flat = inputs.transpose(0, 2, 3, 1).reshape(-1, c)

        if training:
            mean = np.mean(x_flat, axis=0)
            var = np.var(x_flat, axis=0)

            if self.running_mean is None:
                self.running_mean = mean
                self.running_var = var
            else:
                self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
                self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            x_norm = (x_flat - mean) / np.sqrt(var + self.eps)
            self.cache = {'x_flat': x_flat, 'mean': mean, 'var': var,
                          'x_norm': x_norm, 'gamma': self.params["gamma"], 'eps': self.eps}
        else:
            x_norm = (x_flat - self.running_mean) / np.sqrt(self.running_var + self.eps)

        out = self.params["gamma"] * x_norm + self.params["beta"]
        out = out.reshape(batch_size, h, w, c).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        batch_size, c, h, w = dout.shape
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, c)

        x_flat = self.cache['x_flat']
        mean = self.cache['mean']
        var = self.cache['var']
        x_norm = self.cache['x_norm']
        gamma = self.cache['gamma']
        eps = self.cache['eps']

        N = x_flat.shape[0]

        self.grads["gamma"] = np.sum(dout_flat * x_norm, axis=0)
        self.grads["beta"] = np.sum(dout_flat, axis=0)

        dx_norm = dout_flat * gamma
        dvar = np.sum(dx_norm * (x_flat - mean) * -0.5 * np.power(var + eps, -1.5), axis=0)
        dmean = np.sum(dx_norm * -1 / np.sqrt(var + eps), axis=0) + dvar * np.mean(-2 * (x_flat - mean), axis=0)

        dx = dx_norm / np.sqrt(var + eps) + dvar * 2 * (x_flat - mean) / N + dmean / N
        dinputs = dx.reshape(batch_size, h, w, c).transpose(0, 3, 1, 2)

        return dinputs


class PReLU(Layer):
    """Parametric ReLU activation, per‐channel alpha"""

    def __init__(self, num_parameters=1):
        super().__init__()
        self.num_parameters = num_parameters

    def initialize(self, input_shape):
        # input_shape = (C, H, W)
        # 初始化每个通道的 alpha
        self.params["alpha"] = np.ones(self.num_parameters) * 0.25
        return input_shape

    def forward(self, inputs):
        # inputs: (N, C, H, W)
        self.inputs = inputs
        # reshape alpha to (1, C, 1, 1) 以便广播
        alpha = self.params["alpha"].reshape(1, -1, 1, 1)
        # 正部分保持原值，负部分乘 alpha
        out = np.where(inputs > 0, inputs, inputs * alpha)
        return out

    def backward(self, dout):
        # dout: (N, C, H, W)
        alpha = self.params["alpha"].reshape(1, -1, 1, 1)
        # grad w.r.t. inputs: 正部分通道为 1，负部分为 alpha

        # grad w.r.t. alpha：sum over N, H, W 维度，仅在 inputs <= 0 时贡献
        # 公式：dL/dα_c = sum_{i where x_i_c <=0} dout_i_c * x_i_c
        mask = (self.inputs <= 0)
        grad_alpha = np.sum(dout * self.inputs * mask, axis=(0,2,3)) # 按通道更新

        self.grads["alpha"] = grad_alpha
        dinputs = np.where(self.inputs > 0, dout, dout * alpha)
        return dinputs


class PixelShuffle(Layer):
    """Pixel Shuffle layer for upsampling"""

    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def initialize(self, input_shape):
        c, h, w = input_shape
        if c % (self.scale_factor ** 2) != 0:
            raise ValueError(f"Channel count {c} must be divisible by scale_factor^2")

        output_c = c // (self.scale_factor ** 2)
        output_h = h * self.scale_factor
        output_w = w * self.scale_factor

        return (output_c, output_h, output_w)

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, c, h, w = inputs.shape
        r = self.scale_factor

        output_c = c // (r ** 2)
        output_h = h * r
        output_w = w * r

        x = inputs.reshape(batch_size, output_c, r, r, h, w)
        x = x.transpose(0, 1, 4, 2, 5, 3)
        outputs = x.reshape(batch_size, output_c, output_h, output_w)

        return outputs

    def backward(self, dout):
        batch_size, c, h, w = dout.shape
        r = self.scale_factor

        input_c = c * (r ** 2)
        input_h = h // r
        input_w = w // r

        x = dout.reshape(batch_size, c, input_h, r, input_w, r)
        x = x.transpose(0, 1, 3, 5, 2, 4)
        dinputs = x.reshape(batch_size, input_c, input_h, input_w)

        return dinputs


class Add(Layer):
    """Addition layer for residual connections"""

    def forward(self, inputs):
        self.inputs = inputs
        return inputs[0] + inputs[1]

    def backward(self, dout):
        return [dout, dout]


class ResidualBlock:
    """Residual block with two conv layers and skip connection"""

    def __init__(self, channels):
        self.layers = {}

        # First conv block
        self.layers["conv1"] = Conv2D(channels, kernel_size=3, padding=1)
        self.layers["bn1"] = BatchNorm2D(channels)
        self.layers["prelu"] = PReLU(channels)

        # Second conv block
        self.layers["conv2"] = Conv2D(channels, kernel_size=3, padding=1)
        self.layers["bn2"] = BatchNorm2D(channels)

        # Skip connection
        self.layers["add"] = Add()

    def initialize(self, input_shape):
        x_shape = input_shape

        # Initialize each layer
        self.layers["conv1"].initialize(x_shape)
        self.layers["bn1"].initialize(self.layers["conv1"].initialize(x_shape))
        self.layers["prelu"].initialize(self.layers["bn1"].initialize(self.layers["conv1"].initialize(x_shape)))

        self.layers["conv2"].initialize(
            self.layers["prelu"].initialize(self.layers["bn1"].initialize(self.layers["conv1"].initialize(x_shape))))
        self.layers["bn2"].initialize(self.layers["conv2"].initialize(
            self.layers["prelu"].initialize(self.layers["bn1"].initialize(self.layers["conv1"].initialize(x_shape)))))

        return x_shape

    def forward(self, inputs, training=True):
        # Forward pass through layers
        x = self.layers["conv1"].forward(inputs)
        x = self.layers["bn1"].forward(x, training=training)
        x = self.layers["prelu"].forward(x)

        x = self.layers["conv2"].forward(x)
        x = self.layers["bn2"].forward(x, training=training)

        # Add skip connection
        x = self.layers["add"].forward([x, inputs])

        return x

    def backward(self, dout):
        # Backward pass through layers
        [dx, dinputs] = self.layers["add"].backward(dout)

        dx = self.layers["bn2"].backward(dx)
        dx = self.layers["conv2"].backward(dx)

        dx = self.layers["prelu"].backward(dx)
        dx = self.layers["bn1"].backward(dx)
        dx = self.layers["conv1"].backward(dx)

        # Add gradient from skip connection
        dx = dx + dinputs

        return dx

    def get_params_and_grads(self):
        params = {}
        grads = {}

        for layer_name, layer in self.layers.items():
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for param_name, param in layer.params.items():
                    params[f"{layer_name}_{param_name}"] = param
                    grads[f"{layer_name}_{param_name}"] = layer.grads.get(param_name, None)

        return params, grads


class SimpleSRResNet:
    """Simplified Super Resolution Residual Network"""

    def __init__(self, scale_factor=3, num_residual_blocks=8):
        self.scale_factor = scale_factor
        self.num_blocks = num_residual_blocks
        self.layers = {}
        self.residual_blocks = []

        # Input conv
        self.layers["conv_input"] = Conv2D(64, kernel_size=3, padding=1)
        self.layers["prelu_input"] = PReLU(64)

        # Residual blocks
        for i in range(self.num_blocks):
            self.residual_blocks.append(ResidualBlock(64))

        # Post-residual conv
        self.layers["conv_mid"] = Conv2D(64, kernel_size=3, padding=1)
        self.layers["bn_mid"] = BatchNorm2D(64)
        self.layers["add_global"] = Add()

        # Upsampling
        self.layers["conv_up"] = Conv2D(self.scale_factor ** 2 * 64, kernel_size=3, padding=1)
        self.layers["pixel_shuffle"] = PixelShuffle(scale_factor)
        self.layers["prelu_up"] = PReLU(64)

        # Output conv
        self.layers["conv_output"] = Conv2D(3, kernel_size=3, padding=1)

    def initialize(self, input_shape):
        c, h, w = input_shape

        # Initialize feature extraction
        feature_shape = self.layers["prelu_input"].initialize(
            self.layers["conv_input"].initialize((c, h, w))
        )

        # Save feature shape for global residual connection
        self.feature_shape = feature_shape

        # Initialize residual blocks
        x_shape = feature_shape
        for block in self.residual_blocks:
            x_shape = block.initialize(x_shape)

        # Initialize post-residual layers
        self.layers["conv_mid"].initialize(x_shape)
        self.layers["bn_mid"].initialize(self.layers["conv_mid"].initialize(x_shape))

        # Initialize upsampling layers
        self.layers["conv_up"].initialize(feature_shape)
        self.layers["pixel_shuffle"].initialize(self.layers["conv_up"].initialize(feature_shape))
        self.layers["prelu_up"].initialize(self.layers["pixel_shuffle"].initialize(
            self.layers["conv_up"].initialize(feature_shape))
        )

        # Initialize output layer
        output_shape = self.layers["conv_output"].initialize(
            self.layers["prelu_up"].initialize(
                self.layers["pixel_shuffle"].initialize(
                    self.layers["conv_up"].initialize(feature_shape)
                )
            )
        )

        # return (batch_size, *output_shape)

        return output_shape  # 直接返回输出形状，无需包含批次大小

    def forward(self, inputs, training=True):
        # Feature extraction
        x = self.layers["conv_input"].forward(inputs)
        x = self.layers["prelu_input"].forward(x)

        # Save for global residual connection
        feature_out = x

        # Residual blocks
        for block in self.residual_blocks:
            x = block.forward(x, training=training)

        # Post-residual processing
        x = self.layers["conv_mid"].forward(x)
        x = self.layers["bn_mid"].forward(x, training=training)

        # Global residual connection
        x = self.layers["add_global"].forward([x, feature_out])

        # Upsampling
        x = self.layers["conv_up"].forward(x)
        x = self.layers["pixel_shuffle"].forward(x)
        x = self.layers["prelu_up"].forward(x)

        # Output
        x = self.layers["conv_output"].forward(x)

        # Map to [0, 1] range
        # sigmoid可能导致梯度饱和（尤其是当输入较大时），抑制模型学习，且输出严格在（0,1）
        # 而真实值在[0,1]，可能导致psnr计算偏差
        # x = 1.0 / (1.0 + np.exp(-x))  # Simple sigmoid
        x = np.clip(x, 0.0, 1.0)  # 替换 Sigmoid

        self.output = x

        return x

    def backward(self, dout):
        # Output gradient
        dx = dout * self.output * (1 - self.output)  # Sigmoid gradient
        dx = self.layers["conv_output"].backward(dx)

        # Upsampling gradient
        dx = self.layers["prelu_up"].backward(dx)
        dx = self.layers["pixel_shuffle"].backward(dx)
        dx = self.layers["conv_up"].backward(dx)

        # Global residual gradient
        [dx, dfeature_out] = self.layers["add_global"].backward(dx)

        # Post-residual gradient
        dx = self.layers["bn_mid"].backward(dx)
        dx = self.layers["conv_mid"].backward(dx)

        # Residual blocks gradient
        for block in reversed(self.residual_blocks):
            dx = block.backward(dx)

        # Feature extraction gradient
        dx = self.layers["prelu_input"].backward(dx + dfeature_out)
        dx = self.layers["conv_input"].backward(dx)

        return dx

    def get_params_and_grads(self):
        params = {}
        grads = {}

        # Get base layer parameters
        for layer_name, layer in self.layers.items():
            if hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for param_name, param in layer.params.items():
                    params[f"{layer_name}_{param_name}"] = param
                    grads[f"{layer_name}_{param_name}"] = layer.grads.get(param_name, None)

        # Get residual block parameters
        for i, block in enumerate(self.residual_blocks):
            block_params, block_grads = block.get_params_and_grads()
            params.update({f"resblock{i}_{k}": v for k, v in block_params.items()})
            grads.update({f"resblock{i}_{k}": v for k, v in block_grads.items()})

        return params, grads

    def save_weights(self, filename):
        """Save model weights and BN running stats to file"""
        params, _ = self.get_params_and_grads()
        extra = {}
        # Save BatchNorm running stats
        for layer_name, layer in self.layers.items():
            if isinstance(layer, BatchNorm2D):
                extra[f"{layer_name}_running_mean"] = layer.running_mean
                extra[f"{layer_name}_running_var"] = layer.running_var
        for i, block in enumerate(self.residual_blocks):
            for lname, l in block.layers.items():
                if isinstance(l, BatchNorm2D):
                    extra[f"resblock{i}_{lname}_running_mean"] = l.running_mean
                    extra[f"resblock{i}_{lname}_running_var"] = l.running_var
        np.savez(filename, **params, **extra)

    def load_weights(self, filename):
        """Load model weights and BN running stats from file"""
        params = np.load(filename)
        for key, value in params.items():
            if "running_mean" in key or "running_var" in key:
                # Handle BN running stats
                parts = key.split("_")
                if parts[0].startswith('resblock'):
                    block_idx = int(parts[0][8:])
                    layer_name = parts[1]
                    if block_idx < len(self.residual_blocks):
                        if "running_mean" in key:
                            self.residual_blocks[block_idx].layers[layer_name].running_mean = value
                        else:
                            self.residual_blocks[block_idx].layers[layer_name].running_var = value
                else:
                    layer_name = parts[0]
                    if layer_name in self.layers:
                        if "running_mean" in key:
                            self.layers[layer_name].running_mean = value
                        else:
                            self.layers[layer_name].running_var = value
            else:
                # Normal param
                parts = key.split("_")
                if parts[0].startswith('resblock'):
                    block_idx = int(parts[0][8:])
                    layer_name = parts[1]
                    param_name = parts[2]
                    if block_idx < len(self.residual_blocks):
                        self.residual_blocks[block_idx].layers[layer_name].params[param_name] = value
                else:
                    layer_name = parts[0]
                    param_name = parts[1]
                    if layer_name in self.layers:
                        self.layers[layer_name].params[param_name] = value