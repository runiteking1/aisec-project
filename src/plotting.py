import os

import numpy as np
from matplotlib import pyplot as plt


def plot_training_metrics(batch_accuracies, batch_losses, test_accuracies, batches_per_epoch, output_dir):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(range(1, len(batch_losses) + 1), batch_losses, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(range(1, len(batch_accuracies) + 1), batch_accuracies, color=color, label='Train Accuracy')
    ax2.plot(
        np.arange(1, len(test_accuracies) + 1) * batches_per_epoch,
        test_accuracies, 'o-', color='g', linewidth=4.0, markersize=10.0,
        label='Test Accuracy',
    )
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Training Progress: Loss and Accuracy per Batch')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    plot_path = os.path.join(output_dir, 'training_metrics.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
