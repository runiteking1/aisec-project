from functools import partial

import jax.numpy as jnp
import jax
import os

import numpy as np
import optax
import hydra
from jax import Array
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from flax.training.train_state import TrainState

from src.models import CNN
from typing import Tuple, Any
import orbax.checkpoint as ocp
import logging

log = logging.getLogger(__name__)


@partial(jax.jit, static_argnums=(2,))
def train_step(state: TrainState, batch: Tuple[jnp.ndarray, jnp.ndarray], num_classes: int,
               gauss_newton: float = None) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    """
    Performs a single training step using a TrainState object.

    Args:
        state: The current TrainState, containing model parameters and optimizer state.
        batch: A tuple (images, labels) for a single batch.
        num_classes: The number of classes in the dataset.
        gauss_newton: if defined, uses that as the regularizer for Gauss Newton's (aka Levenberg-Marquardt)

    Returns:
        A tuple of (updated_state, loss, accuracy).
    """

    @jax.jit
    def cross_entropy_loss(params, batch: jnp.ndarray) -> tuple[Array, tuple[Any, Array]]:
        """
        Computes the cross-entropy loss and accuracy for a given batch.

        Args:
            params: The model parameters.
            batch: A tuple (images, labels) for a single batch.

        Returns:
            A tuple of (loss, accuracy).
        """
        images, labels = batch
        logits = state.apply_fn({'params': params}, images)
        one_hot_labels = jax.nn.one_hot(labels, num_classes=num_classes)
        loss = -jnp.sum(one_hot_labels * jax.nn.log_softmax(logits)) / len(images)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        return loss, (logits, accuracy)

    # We use jax.value_and_grad to compute both the loss and the gradients in a single step.
    (loss, (logits, accuracy)), grads = jax.value_and_grad(
        cross_entropy_loss, has_aux=True
    )(state.params, batch)

    # This is not that optimized; but it's okay
    if gauss_newton is not None:
        bsz = len(batch[0])
        jacobian = jax.jacobian(
            lambda p: state.apply_fn({'params': p}, batch[0])
        )(state.params)
        flattened, _ = jax.tree.flatten(
            jacobian
        )

        reshaped_leaves = [leaf.reshape(
            bsz * num_classes, 1, -1) for leaf in flattened]
        aggregated_array = jnp.concatenate(reshaped_leaves, axis=-1)
        jacobian = aggregated_array.reshape((bsz * num_classes, -1))

        # Generalized Gauss-Newton is J^THJ
        probs = jax.nn.softmax(logits, axis=-1).flatten()
        diag_probs = jnp.diag(probs)
        outer_probs = jnp.outer(probs, probs)
        h_mat = diag_probs - outer_probs

        flattened, back = jax.flatten_util.ravel_pytree(grads)
        altered = jnp.linalg.solve(jacobian.T @ h_mat @ jacobian + 0.1 * jnp.eye(jacobian.shape[1]), flattened)
        grads = back(altered)

    # Compute the parameter updates using the optimizer and apply them
    new_state = state.apply_gradients(grads=grads)

    return new_state, loss, accuracy


def get_datasets():
    """
    Loads and prepares the MNIST dataset using the 'datasets' library,
    saving the processed data and loading it if it already exists.
    """
    processed_train_path = 'processed_mnist_train.npy'
    processed_test_path = 'processed_mnist_test.npy'

    # Check if the processed dataset files exist
    if os.path.exists(processed_train_path) and os.path.exists(processed_test_path):
        log.info("✅ Processed datasets found. Loading from disk...")
        train_data = jnp.load(processed_train_path, allow_pickle=True).item()
        test_data = jnp.load(processed_test_path, allow_pickle=True).item()
        return {
            'train': train_data,
            'test': test_data
        }

    # If not found, process and save the dataset
    log.info("⏳ Processed datasets not found. Processing and saving...")
    mnist_ds = load_dataset('mnist')

    def preprocess(batch):
        images = jnp.array(batch['image']) / 255.0
        images = jnp.expand_dims(images, axis=-1)
        labels = jnp.array(batch['label'])
        return {'image': images, 'label': labels}

    mnist_ds = mnist_ds.map(preprocess, batched=True)
    train_ds = mnist_ds['train']
    test_ds = mnist_ds['test']

    train_datasets = {
        'image': jnp.array(train_ds['image']),
        'label': jnp.array(train_ds['label'])
    }
    test_datasets = {
        'image': jnp.array(test_ds['image']),
        'label': jnp.array(test_ds['label'])
    }

    # Save the processed datasets as .npy files
    jnp.save(processed_train_path, train_datasets)
    jnp.save(processed_test_path, test_datasets)

    log.info("💾 Processed datasets saved successfully.")
    return {
        'train': train_datasets,
        'test': test_datasets
    }


def get_cifar10_datasets():
    """
    Loads and prepares the CIFAR-10 dataset using the 'datasets' library,
    saving the processed data and loading it if it already exists.
    """
    processed_train_path = 'processed_cifar10_train.npy'
    processed_test_path = 'processed_cifar10_test.npy'

    # Check if the processed dataset files exist
    if os.path.exists(processed_train_path) and os.path.exists(processed_test_path):
        log.info("✅ Processed CIFAR-10 datasets found. Loading from disk...")
        train_data = jnp.load(processed_train_path, allow_pickle=True).item()
        test_data = jnp.load(processed_test_path, allow_pickle=True).item()
        return {
            'train': train_data,
            'test': test_data
        }

    # If not found, process and save the dataset
    log.info("⏳ Processed CIFAR-10 datasets not found. Processing and saving...")
    cifar10_ds = load_dataset('cifar10')

    def preprocess(batch):
        # CIFAR-10 images are 32x32 with 3 color channels (RGB), so no need to add an axis.
        # Normalize pixel values to be between 0 and 1.
        images = jnp.array(batch['img']) / 255.0
        labels = jnp.array(batch['label'])
        return {'image': images, 'label': labels}

    cifar10_ds = cifar10_ds.map(preprocess, batched=True)
    train_ds = cifar10_ds['train']
    test_ds = cifar10_ds['test']

    train_datasets = {
        'image': jnp.array(train_ds['image']),
        'label': jnp.array(train_ds['label'])
    }
    test_datasets = {
        'image': jnp.array(test_ds['image']),
        'label': jnp.array(test_ds['label'])
    }

    # Save the processed datasets as .npy files
    jnp.save(processed_train_path, train_datasets)
    jnp.save(processed_test_path, test_datasets)

    log.info("💾 Processed CIFAR-10 datasets saved successfully.")
    return {
        'train': train_datasets,
        'test': test_datasets
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    """
    Main training function decorated with Hydra.

    Args:
        cfg: The Hydra configuration object.
    """
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"Output directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    log.info(f"Using {jax.default_backend()}")

    # Get a JAX PRNGKey for reproducibility
    key = jax.random.PRNGKey(cfg.seed)

    # Instantiate the model and optimizer based on the config
    model = CNN(
        features_per_layer=tuple(cfg.model.features_per_layer),
        kernel_size=tuple(cfg.model.kernel_size),
        dense_features=tuple(cfg.model.dense_features),
        num_classes=cfg.model.num_classes
    )

    # Dynamically select the optimizer based on the config file
    gn_param = None
    if cfg.training.optimizer == 'adam':
        optimizer = optax.adam(learning_rate=cfg.training.learning_rate)
    elif cfg.training.optimizer == 'sgd':
        optimizer = optax.sgd(learning_rate=cfg.training.learning_rate)
    elif cfg.training.optimizer == 'gn':
        optimizer = optax.sgd(learning_rate=cfg.training.learning_rate)
        gn_param = cfg.training.gn_param
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}.")

    # Load the datasets
    if cfg.data.name == 'cifar10':
        datasets = get_cifar10_datasets()
    elif cfg.data.name == 'mnist':
        datasets = get_datasets()
    else:
        raise ValueError(f"Unknown dataset: {cfg.data.name}.")

    train_ds = datasets['train']
    test_ds = datasets['test']

    # Create a dummy input to initialize the model parameters
    dummy_input = jnp.ones(
        (1, cfg.data.image_size, cfg.data.image_size, cfg.data.channels)
    )

    # Initialize the model and create the TrainState object
    params = model.init(key, dummy_input)['params']
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    log.info("Starting training...")
    log.info(f"Total number of parameters: {param_count}")

    # Get number of samples and batch size from config
    num_train_samples = len(train_ds['image'])
    num_test_samples = len(test_ds['image'])
    batch_size = cfg.training.batch_size

    test_accuracies = np.zeros(cfg.training.epochs)
    train_losses = np.zeros(cfg.training.epochs)
    for epoch in range(cfg.training.epochs):
        # Split the key for shuffling
        key, subkey = jax.random.split(key)

        # Get a new random permutation of indices for this epoch
        shuffled_indices = jax.random.permutation(subkey, num_train_samples)

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        epoch_test_accuracy = 0.0
        num_batches = 0

        # Iterate through the dataset in batches using the shuffled indices
        for i in range(0, num_train_samples, batch_size):
            batch_indices = shuffled_indices[i:i + batch_size]
            images = train_ds['image'][batch_indices]
            labels = train_ds['label'][batch_indices]
            batch = (images, labels)

            # Call the JIT-compiled train step
            state, loss, accuracy = train_step(state, batch, 10, gauss_newton=gn_param)

            epoch_loss += loss
            epoch_accuracy += accuracy
            num_batches += 1

        # Get test accuracies
        num_test_batches = num_test_samples // batch_size
        for i in range(num_test_batches + 1):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_test_samples)

            image_batch = test_ds['image'][start_idx:end_idx]
            label_batch = test_ds['label'][start_idx:end_idx]

            # Get the model's predictions on the original, clean images
            logits = state.apply_fn({'params': state.params}, image_batch)
            predictions = jnp.argmax(logits, axis=-1)

            epoch_test_accuracy += jnp.sum(predictions == label_batch)

        avg_loss = epoch_loss / num_batches
        avg_accuracy = epoch_accuracy / num_batches

        test_accuracies[epoch] = epoch_test_accuracy / num_test_samples
        train_losses[epoch] = epoch_loss / avg_loss

        log.info(f"Epoch {epoch + 1}/{cfg.training.epochs} | "
                 f"Loss: {avg_loss:.4f} | "
                 f"Train Accuracy: {avg_accuracy:.4f} | "
                 f"Test Accuracy: {test_accuracies[epoch] :.4f}")

    log.info("Training finished.")

    # Instantiate the Orbax Checkpointer. The PyTreeCheckpointer is a general-purpose choice.
    checkpointer = ocp.PyTreeCheckpointer()

    # Convert the Hydra config to a standard Python dictionary to save it
    # with the checkpoint. The resolve=True makes sure all variables are
    # evaluated.
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Create a dictionary to save, which now includes both the state and the config.
    target_to_save = {
        'state': state,
        'config': config_dict
    }

    # Convert the relative path to an absolute path
    checkpoint_path = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, 'checkpoint', 'model')

    # Use Orbax's save method with the absolute path.
    checkpointer.save(checkpoint_path, item=target_to_save)
    log.info("Model and config saved successfully with Orbax!")


# Run the training script
if __name__ == "__main__":
    train()
