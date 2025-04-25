import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import os

def plot_generated_images(generated_images, epoch, output_dir='outputs', examples=25, dim=(5, 5), figsize=(10, 10)):
    """
    Save a grid of generated images to the specified directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    file_path = os.path.join(output_dir, f"generated_epoch_{epoch}.png")
    plt.savefig(file_path)
    plt.close()

def normalize_images(images):
    """
    Normalize image pixel values from [0, 255] to [-1, 1].
    """
    return (images.astype('float32') - 127.5) / 127.5

def denormalize_images(images):
    """
    Convert normalized images back to [0, 255] pixel range.
    """
    return (images * 127.5 + 127.5).astype(np.uint8)
