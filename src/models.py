from flax import linen as nn


class CNN(nn.Module):
    features_per_layer: tuple
    kernel_size: tuple
    dense_features: tuple
    num_classes: int

    @nn.compact
    def __call__(self, x):
        # A simple CNN architecture that is now configurable via the config file.
        # Iterate through the convolutional layers
        for features in self.features_per_layer:
            x = nn.Conv(features=features, kernel_size=self.kernel_size)(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = x.reshape((x.shape[0], -1))  # Flatten the tensor

        # Iterate through the dense layers
        for features in self.dense_features:
            x = nn.Dense(features=features)(x)
            x = nn.relu(x)

        # The final dense layer maps to the number of classes
        x = nn.Dense(features=self.num_classes)(x)
        return x
