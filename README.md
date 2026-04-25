# Deep Learning with PyTorch: DCGAN for MNIST Handwritten Digit Generation

A compact but fully functional implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) in PyTorch for synthesizing handwritten digit images from the MNIST dataset. The notebook follows the standard adversarial learning setup: a **Generator** learns to map Gaussian noise to digit-like images, while a **Discriminator** learns to classify real MNIST images against generated samples.

## Overview

This project demonstrates the full end-to-end workflow of training a GAN from scratch with PyTorch, including dataset preparation, batch loading, network design, adversarial loss formulation, alternating optimization, and image synthesis. The implementation is centered on MNIST digits, making it a clear reference for understanding how convolutional layers, transposed convolutions, normalization, and adversarial training interact in a generative vision pipeline.

The notebook uses:

- `torch` for model definition and training
- `torchvision` for the MNIST dataset and image transforms
- `torch.utils.data.DataLoader` for mini-batch iteration
- `matplotlib` for visualization
- `torchsummary` for architecture inspection
- `tqdm` for progress tracking

## Technical Objectives

The implementation is built around four core goals:

1. Load MNIST into mini-batches with a `DataLoader`.
2. Construct a convolutional Discriminator that distinguishes real images from fake ones.
3. Construct a Generator that transforms latent noise into 28×28 grayscale digits.
4. Train both networks in an alternating adversarial loop.

## Dataset and Preprocessing

The notebook downloads the MNIST training split and applies a lightweight augmentation pipeline:

- `RandomRotation((-20, +20))`
- `ToTensor()`

The rotation augmentation slightly increases the variation in the real training distribution, which can help the discriminator generalize better and makes the adversarial task more robust. After transformation, each image is represented as a tensor of shape `(1, 28, 28)`.

The training dataset contains **60,000 images**, and the batch loader is configured with:

- `batch_size = 128`
- `shuffle = True`

This produces **469 mini-batches** per epoch.

## Model Architecture

### Discriminator

The Discriminator is a compact convolutional classifier that progressively compresses the input image into a single authenticity logit.

Architecture:

- Input: `(batch_size, 1, 28, 28)`
- Conv2d `1 → 16`, kernel `3×3`, stride `2`
- BatchNorm2d
- LeakyReLU
- Conv2d `16 → 32`, kernel `5×5`, stride `2`
- BatchNorm2d
- LeakyReLU
- Conv2d `32 → 64`, kernel `5×5`, stride `2`
- BatchNorm2d
- LeakyReLU
- Flatten
- Linear `64 → 1`

The network outputs a single scalar per image, which is interpreted as the real/fake score used by `BCEWithLogitsLoss`.

Total parameters: **64,545**

### Generator

The Generator maps a latent noise vector into a 28×28 grayscale image using transposed convolutions.

Architecture:

- Input: noise vector of size `z_dim = 64`
- Reshape to `(batch_size, 64, 1, 1)`
- ConvTranspose2d `64 → 256`, kernel `3×3`, stride `2`
- BatchNorm2d
- ReLU
- ConvTranspose2d `256 → 128`, kernel `4×4`, stride `1`
- BatchNorm2d
- ReLU
- ConvTranspose2d `128 → 64`, kernel `3×3`, stride `2`
- BatchNorm2d
- ReLU
- ConvTranspose2d `64 → 1`, kernel `4×4`, stride `2`
- Tanh

The final `Tanh` constrains the output to `[-1, 1]`, which is standard for GAN image generation when working with normalized image tensors.

Total parameters: **747,841**

## Weight Initialization

Both networks are initialized with DCGAN-style normal initialization:

- Convolutional and transposed convolutional weights: `Normal(0, 0.02)`
- BatchNorm2d weights: `Normal(0, 0.02)`
- BatchNorm2d biases: `0`

This initialization helps stabilize early GAN training and keeps activations well-scaled across the generator and discriminator.

## Loss Functions and Optimizers

The training objective uses binary cross-entropy with logits:

- `real_loss(pred)`: compares discriminator outputs on real images against labels of `1`
- `fake_loss(pred)`: compares discriminator outputs on generated images against labels of `0`

Both models are optimized with Adam using the following hyperparameters:

- learning rate: `0.0002`
- `beta_1 = 0.5`
- `beta_2 = 0.99`

These settings are commonly used in DCGAN-style training because they improve stability relative to the default Adam parameters.

## Project Workflow

The notebook follows a clear adversarial learning pipeline:

1. **Prepare the data**  
   MNIST is downloaded, transformed, and loaded into shuffled mini-batches.

2. **Inspect the input distribution**  
   A sample image is visualized to confirm that the dataset is loaded correctly.

3. **Define the Discriminator**  
   A CNN-based classifier is created to output a real/fake logit for each image.

4. **Define the Generator**  
   A transposed-convolution network is built to map latent noise into digit images.

5. **Initialize the weights**  
   Both models receive a DCGAN-compatible parameter initialization.

6. **Create the adversarial objectives**  
   Real and fake targets are defined using `BCEWithLogitsLoss`.

7. **Train the Discriminator**  
   For each batch, the Discriminator is updated using both generated images and real MNIST images.

8. **Train the Generator**  
   The Generator is updated to maximize the Discriminator’s belief that fake images are real.

9. **Monitor outputs**  
   At the end of each epoch, losses are printed and generated samples are visualized.

This alternating optimization scheme is the core of GAN training: the Discriminator learns to detect fakes, while the Generator learns to produce samples that are increasingly difficult to distinguish from real digits.

## Results

Training was run for **20 epochs** on MNIST. The logged losses show a gradual stabilization of the adversarial game:

- Epoch 1: `D_loss ≈ 0.6861`, `G_loss ≈ 0.6890`
- Epoch 20: `D_loss ≈ 0.6450`, `G_loss ≈ 0.8230`

The generated samples evolve from noise-like blobs in the early epochs to more digit-shaped structures as training progresses. Because the Generator ends with `Tanh`, the image tensors are in the normalized range `[-1, 1]`; when displayed directly with Matplotlib, the notebook emits clipping warnings, which is expected for unscaled GAN outputs.

Qualitatively, the final samples demonstrate that the model has learned the MNIST manifold sufficiently to synthesize handwritten digit-like images rather than random texture.

## Implementation Notes

- The notebook uses `device = 'cuda'`, so it expects a CUDA-capable environment.
- The latent space dimensionality is fixed at `64`.
- The training loop updates the Discriminator and Generator in separate optimization steps for each mini-batch.
- The Generator is trained against the Discriminator’s real-label objective, which is the standard non-saturating GAN formulation used in many introductory DCGAN implementations.
- Image visualization uses a reusable helper that creates a grid of samples from a batch tensor.
