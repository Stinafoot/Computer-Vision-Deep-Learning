# Computer Vision Deep Learning (YOLO · GAN · Autoencoder · ViT · CLIP)

Three deep learning computer vision questions covering transfer learning for detection and segmentation, generative modeling with GANs and Autoencoders, and fine-tuned vs. zero-shot classification with Vision Transformers and CLIP.

![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red) ![YOLO](https://img.shields.io/badge/YOLO-v26-darkblue) ![CLIP](https://img.shields.io/badge/OpenAI-CLIP-lightblue)

---

## Q1 - Transfer Learning: Detection & Segmentation with YOLO 

**File:** `TransferLearning.ipynb`  
**Dataset:** [Oxford-IIIT Pet Dataset](https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset)

Applies two pretrained YOLO26 models to the same 5 pet images and saves side-by-side comparisons.

| Model | File | Output |
|-------|------|--------|
| Detection | `yolo26n.pt` | Bounding boxes + confidence scores |
| Segmentation | `yolo26n-seg.pt` | Pixel-level instance masks |

**Tasks:**
1. Download the Oxford-IIIT Pet Dataset (cats and dogs)
2. Run detection on 5 images - visualize bounding boxes and confidence scores
3. Run segmentation on the same 5 images - visualize pixel-level instance masks
4. Save side-by-side annotated output images for each of the 5 images
5. Analyze model performance: how well does YOLO26 generalize to pet images?

---

## Q2 - GAN and Autoencoder on Animal Faces 

**File:** `GANandAutoencoder.ipynb`  
**Dataset:** [Animal Faces (Kaggle)](https://www.kaggle.com/datasets/andrewmvd/animal-faces) - cats, dogs, wild animals

### GAN

```
Random noise (latent vector)
       │
  Generator (CNN decoder)
       │
  Fake image
       │
  Discriminator (CNN encoder) ← Real images
       │
  BCELoss (adversarial)
```

- Generator: upsampling CNN from random noise to 64×64 RGB image
- Discriminator: downsampling CNN outputting real/fake probability
- Loss: `BCELoss` for both networks
- Optimizer: `Adam`
- Training: ≥10 epochs
- **Output:** 5 generated fake animal face images

### Autoencoder

```
Image (64×64×3)
       │
  Encoder → 16-D latent vector
       │
  Decoder → Reconstructed image
       │
  MSELoss
```

- Bottleneck: 16-dimensional latent space
- Loss: `MSELoss` (pixel-wise reconstruction)
- **Output:** 5 original images alongside reconstructions + final reconstruction loss

### Analysis

Discusses the core conceptual difference:
- **GAN:** adversarial loss → perceptually sharp but potentially unfaithful
- **Autoencoder:** reconstruction loss → faithful but blurry (MSE averages over plausible outputs)

---

## Q3 - Vision Transformer (ViT) vs. CLIP Zero-Shot 

**File:** `VisionTransformerandCLIP.ipynb`  
**Dataset:** [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) - 25,000 images, 6 scene categories

### Fine-tuned ViT

```python
model = torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

# Freeze all pretrained weights
for param in model.parameters():
    param.requires_grad = False

# Replace classification head
model.heads.head = nn.Linear(num_features, 6)
```

- Training: ≥5 epochs, `CrossEntropyLoss`, `Adam`
- Only the new classification head is trained

### CLIP Zero-Shot

```python
# Text prompts per class
prompts = [
    "a photo of a building",
    "a photo of a forest",
    "a photo of a glacier",
    "a photo of a mountain",
    "a photo of a sea",
    "a photo of a street",
]
# Compare text embeddings against image embeddings → argmax = predicted class
```

No training on the target dataset - pure zero-shot inference.

### Evaluation

- Accuracy on test set for both models
- Visualization: 5 test images with true label, ViT prediction, and CLIP prediction displayed together

### Key Discussion Points

- ViT (fine-tuned) vs. CLIP (zero-shot): accuracy trade-off
- Value of task-specific training vs. large-scale contrastive pretraining
- Which approach is more practical with limited labeled data?

---

## Requirements

```bash
pip install torch torchvision transformers pillow matplotlib
# For CLIP:
pip install git+https://github.com/openai/CLIP.git
# For YOLO:
pip install ultralytics
```

## Usage

```bash
# Download datasets from Kaggle links above
jupyter notebook TransferLearning.ipynb      # Q1
jupyter notebook GANandAutoencoder.ipynb     # Q2
jupyter notebook VisionTransformerandCLIP.ipynb  # Q3
```

---

## Files

| File | Description |
|------|-------------|
| `TransferLearning.ipynb` | Q1: YOLO26 detection + segmentation |
| `GANandAutoencoder.ipynb` | Q2: GAN + Autoencoder training and visualization |
| `VisionTransformerandCLIP.ipynb` | Q3: ViT fine-tuning + CLIP zero-shot |
