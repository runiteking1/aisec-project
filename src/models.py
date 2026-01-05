from flax import linen as nn
import jax.numpy as jnp


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


class PatchEmbedding(nn.Module):
    """Extracts patches and projects to embedding dimension."""
    patch_size: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        # x: (batch, height, width, channels)
        # Use Conv2D to extract patches and project
        # kernel_size=patch_size, strides=patch_size acts as patch extraction
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding='VALID',
            name='patch_embedding'
        )(x)
        # Output: (batch, num_patches_h, num_patches_w, hidden_dim)

        # Flatten spatial dimensions
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1, self.hidden_dim)
        # Output: (batch, num_patches, hidden_dim)

        return x


class MLP(nn.Module):
    """Two-layer MLP with GELU activation."""
    mlp_dim: int
    hidden_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        x = nn.Dense(features=self.mlp_dim, name='fc1')(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        x = nn.Dense(features=self.hidden_dim, name='fc2')(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-norm architecture."""
    hidden_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # Multi-head self-attention block
        attn_output = nn.LayerNorm(name='ln1')(x)
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            name='attn'
        )(attn_output, attn_output)
        x = x + attn_output  # Residual connection

        # MLP block
        mlp_output = nn.LayerNorm(name='ln2')(x)
        mlp_output = MLP(
            mlp_dim=self.mlp_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            name='mlp'
        )(mlp_output, deterministic=deterministic)
        x = x + mlp_output  # Residual connection

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification."""
    patch_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float
    num_classes: int

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        # x: (batch, height, width, channels)
        batch_size = x.shape[0]

        # 1. Patch embedding
        x = PatchEmbedding(
            patch_size=self.patch_size,
            hidden_dim=self.hidden_dim,
            name='patch_embed'
        )(x)
        # x: (batch, num_patches, hidden_dim)

        num_patches = x.shape[1]

        # 2. Add CLS token
        cls_token = self.param(
            'cls_token',
            nn.initializers.zeros,
            (1, 1, self.hidden_dim)
        )
        cls_token = jnp.tile(cls_token, (batch_size, 1, 1))
        x = jnp.concatenate([cls_token, x], axis=1)
        # x: (batch, num_patches + 1, hidden_dim)

        # 3. Add positional embeddings
        pos_embedding = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (1, num_patches + 1, self.hidden_dim)
        )
        x = x + pos_embedding

        # Apply dropout to embeddings
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

        # 4. Transformer blocks
        for i in range(self.num_layers):
            x = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                name=f'transformer_block_{i}'
            )(x, deterministic=deterministic)

        # 5. Classification head
        # Extract CLS token (first token)
        cls_output = x[:, 0]

        # Final layer norm
        cls_output = nn.LayerNorm(name='ln_final')(cls_output)

        # Project to num_classes
        logits = nn.Dense(features=self.num_classes, name='head')(cls_output)

        return logits
