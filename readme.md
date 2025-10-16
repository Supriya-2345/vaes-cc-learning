# 🧠 VAE Training Script for ConCon Dataset

This script launches a **Variational Autoencoder (VAE)** training run on the `concon` dataset with a specified configuration of latent dimension, batch size, number of epochs, and environment ID. It uses a GPU for acceleration and saves all results, outputs, and logs in a structured directory.

---

## 🚀 Quick Start

### 1. Clone or navigate to your project directory.

### 2. Set environment variables and run the script:

```bash
export CUDA_VISIBLE_DEVICES=1  # Select which GPU to use

NUM_EPOCHS=3
LATENT_DIM=28
BATCH_SIZE=64

central_results_folder="/path/"  # Change this to your preferred results path
mkdir -p "$central_results_folder"

RUN_DIR="${central_results_folder}/run_${NUM_EPOCHS}_${LATENT_DIM}_${BATCH_SIZE}"
mkdir -p "$RUN_DIR"

ENV_ID=4
SAVE_NAME="run_${NUM_EPOCHS}_${LATENT_DIM}_${BATCH_SIZE}_env${ENV_ID}"
SAVE_DIR="${central_results_folder}/${SAVE_NAME}"
mkdir -p "$SAVE_DIR"

echo "Using GPU ID: $CUDA_VISIBLE_DEVICES" > "${SAVE_DIR}/gpu_used.txt"


