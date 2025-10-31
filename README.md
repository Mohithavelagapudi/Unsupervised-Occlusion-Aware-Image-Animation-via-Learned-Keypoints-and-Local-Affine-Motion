# 🌀 Unsupervised Occlusion-Aware Image Animation via Learned Keypoints and Local Affine Motion

**An unsupervised deep learning framework for realistic image animation using learned keypoints, local affine motion modeling, and occlusion-aware generation.**

<p align="center">
  <img src="image (28).png" alt="" width="1000"/>
</p>

---

## 🧩 1. Abstract

This project implements an **unsupervised image animation system** that transfers motion from a **driving video** to a **static source image** within the same semantic category (e.g., faces, articulated bodies, gifs).

The method jointly learns:

- A compact **first-order motion representation** consisting of **sparse keypoints** and **local affine transformations**.  
- A **dense motion network** that aggregates local motions into a full **optical flow field** and an **occlusion map**.  
- An **occlusion-aware generator** that warps, filters (using the occlusion mask), and inpaints to produce **temporally coherent frames**.

Training is **self-supervised** — the model reconstructs frames from videos *without any explicit landmark, pose, or segmentation annotations.*

---

## ⚙️ 2. Methodology Overview

The core of the method is a **first-order motion model** that uses a *learned set of keypoints* to represent complex deformations between a source image and a driving video frame.

### 🧮 2.1 Local Affine Transformation

Given a frame \( X \), we estimate each transformation \( T_{X \leftarrow R} \) in the neighborhood of the learned keypoints.

Formally, for \( K \) keypoints \( p_1, \dots, p_K \):

$$
T_{X \leftarrow R}(p) = T_{X \leftarrow R}(p_k) + 
\left(\frac{d}{dp} T_{X \leftarrow R}(p)|_{p=p_k}\right) (p - p_k) + o(\|p - p_k\|)
$$

After computing first-order Taylor expansions for motion from reference pose \( R \) to source \( S \) and driving \( D \), we approximate:

$$
T_{S \leftarrow D}(z) \approx T_{S \leftarrow R}(p_k) + J_k(z - T_{D \leftarrow R}(p_k))
$$

Where the Jacobian \( J_k \) represents the **local affine transformation** at keypoint \( p_k \):

$$
J_k = \left(\frac{d}{dp} T_{S \leftarrow R}(p)|_{p=p_k}\right)
\left(\frac{d}{dp} T_{D \leftarrow R}(p)|_{p=p_k}\right)^{-1}
$$

**Legend**

| Symbol | Meaning |
|---------|----------|
| \( P \) | Point locations in the reference pose space \( R \) |
| \( Z \) | Point locations in the source \( S \) or driving \( D \) space |

---

### 🧠 2.2 Architecture Components

<p align="center">
  <img src="image (29).png" alt="" width="1000"/>
</p>

#### 🧩 Keypoint Detector (KPDetector)
- **Architecture:** Hourglass encoder–decoder  
- **Outputs:** Heatmaps → Keypoint positions via spatial expectation  
- **Optional:** Predicts local Jacobians (2×2 affine transformation per keypoint)

#### 🌊 Dense Motion Network
- Builds Gaussian heatmaps centered at keypoints (for source & driving frames)  
- Constructs **sparse motion fields** using local affine transforms (\( J_k \))  
- Composes them into a **dense deformation field** (optical flow)  
- **Outputs:**
  - Deformation field (dense flow grid)
  - Occlusion map (pixel-wise visibility mask)
  - `sparse_deformed`: per-keypoint warped source features

#### 🎨 Occlusion-Aware Generator
- Encoder–decoder with **residual bottleneck**
- **Warps encoder features** using the dense deformation field
- Applies the **occlusion mask** to filter unreliable regions
- **Decoder** inpaints masked regions to produce coherent, realistic frames

---

## 🧰 3. Data & Preprocessing

### 📦 3.1 Supported Datasets

All datasets are configured via `config/*.yaml` files.

| Dataset | Directory | Frame Shape | Description |
|----------|------------|-------------|--------------|
| **BAIR** | `data/bair` | 256×256×3 | Robotic arm motion |
| **Fashion** | `data/fashion-png` | 256×256×3 | Clothing articulation |
| **Moving GIF** | `data/moving-gif` | 256×256×3 | Diverse synthetic motions |
| **Vox / Taichi** | *(in config folder)* | 256×256×3 | Human articulation / performance |

---

### 🧩 3.2 Sampling Strategy

- Dataset is randomly split into **training** and **test** sets unless predefined folders exist.  
- During training, the `frames_dataset.py` loader samples **two random frames** from the same video:
  - One as the **source frame**
  - One as the **driving frame**

---

## 📁 4. Repository Structure

| File / Directory | Purpose |
|------------------|----------|
| `train.py` | Main training loop orchestrator |
| `animate.py` | Inference script for generating animations |
| `frames_dataset.py` | Loads, pairs, and transforms video frames |
| `augmentation.py` | Data augmentations (flip, crop, color jitter) |
| `modules/*` | Core model components (KPDetector, DenseMotion, Generator, Discriminator, Loss) |
| `config/*.yaml` | Dataset-specific hyperparameter configs |
| `logger.py` | Handles checkpoints, TensorBoard logs, and visualization |
| `sync_batchnorm/*` | Custom synchronized BatchNorm for multi-GPU setups |

---

## 🚀 5. Getting Started

### ⚙️ 5.1 Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/unsupervised-animation.git
cd unsupervised-animation

# 2. Install dependencies
pip install -r requirements.txt
```
### 🏋️‍♀️ 5.2 Training

Run the training script, specifying the dataset config file.
``` bash
# Example training run on the 'fashion' dataset
python train.py --config config/fashion-256.yaml
```
### 🎞️ 5.3 Animation (Inference)

Generate a video animation by transferring motion from a driving video to a source image.
``` bash
python animate.py --config config/fashion-256.yaml \
                  --source_image path/to/source.png \
                  --driving_video path/to/driving.mp4 \
                  --checkpoint path/to/model.pth.tar \
                  --result_video path/to/output.mp4
```
