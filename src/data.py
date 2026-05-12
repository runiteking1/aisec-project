import logging
import os
from random import choices

import jax.numpy as jnp
from datasets import load_dataset

log = logging.getLogger(__name__)


def get_datasets(poison_rate: float = 0.0):
    processed_train_path = 'processed_mnist_train.npy'
    processed_test_path = 'processed_mnist_test.npy'

    if os.path.exists(processed_train_path) and os.path.exists(processed_test_path):
        log.info("Processed datasets found. Loading from disk...")
        train_data = jnp.load(processed_train_path, allow_pickle=True).item()
        test_data = jnp.load(processed_test_path, allow_pickle=True).item()
    else:
        log.info("Processed datasets not found. Processing and saving...")
        mnist_ds = load_dataset('mnist')

        def preprocess(batch):
            images = jnp.array(batch['image']) / 255.0
            images = jnp.expand_dims(images, axis=-1)
            labels = jnp.array(batch['label'])
            return {'image': images, 'label': labels}

        mnist_ds = mnist_ds.map(preprocess, batched=True)
        train_data = {
            'image': jnp.array(mnist_ds['train']['image']),
            'label': jnp.array(mnist_ds['train']['label']),
        }
        test_data = {
            'image': jnp.array(mnist_ds['test']['image']),
            'label': jnp.array(mnist_ds['test']['label']),
        }
        jnp.save(processed_train_path, train_data)
        jnp.save(processed_test_path, test_data)
        log.info("Processed datasets saved.")

    num_to_poison = int(len(train_data['label']) * poison_rate)
    if num_to_poison > 0:
        log.info(f"Poisoning {num_to_poison} ({poison_rate:.2%}) training labels...")
        indices_to_poison = choices(range(len(train_data['label'])), k=num_to_poison)
        for idx in indices_to_poison:
            original_label = train_data['label'][idx]
            all_labels = list(range(10))
            all_labels.remove(original_label)
            train_data['label'] = train_data['label'].at[idx].set(choices(all_labels, k=1)[0])

    return {'train': train_data, 'test': test_data}


def get_cifar10_datasets(num_classes: int = None):
    processed_train_path = 'processed_cifar10_train.npy'
    processed_test_path = 'processed_cifar10_test.npy'

    if os.path.exists(processed_train_path) and os.path.exists(processed_test_path):
        log.info(f"Processed CIFAR-10 datasets found. Loading with {num_classes} classes...")
        train_data = jnp.load(processed_train_path, allow_pickle=True).item()
        test_data = jnp.load(processed_test_path, allow_pickle=True).item()
    else:
        log.info("Processed CIFAR-10 datasets not found. Processing and saving...")
        cifar10_ds = load_dataset('cifar10')

        def preprocess(batch):
            images = jnp.array(batch['img']) / 255.0
            labels = jnp.array(batch['label'])
            return {'image': images, 'label': labels}

        cifar10_ds = cifar10_ds.map(preprocess, batched=True)
        train_data = {
            'image': jnp.array(cifar10_ds['train']['image']),
            'label': jnp.array(cifar10_ds['train']['label']),
        }
        test_data = {
            'image': jnp.array(cifar10_ds['test']['image']),
            'label': jnp.array(cifar10_ds['test']['label']),
        }
        jnp.save(processed_train_path, train_data)
        jnp.save(processed_test_path, test_data)
        log.info("Processed CIFAR-10 datasets saved.")

    train_mask = jnp.array(train_data['label']) < num_classes
    test_mask = jnp.array(test_data['label']) < num_classes
    return {
        'train': {
            'image': jnp.array(train_data['image'])[train_mask],
            'label': jnp.array(train_data['label'])[train_mask],
        },
        'test': {
            'image': jnp.array(test_data['image'])[test_mask],
            'label': jnp.array(test_data['label'])[test_mask],
        },
    }
