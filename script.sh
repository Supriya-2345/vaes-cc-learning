export CUDA_VISIBLE_DEVICES=1 

NUM_EPOCHS=3
LATENT_DIM=28
BATCH_SIZE=64

central_results_folder="/path/"
mkdir -p "$central_results_folder"

RUN_DIR="${central_results_folder}/run_${NUM_EPOCHS}_${LATENT_DIM}_${BATCH_SIZE}"
mkdir -p "$RUN_DIR"

ENV_ID=4
SAVE_NAME="run_${NUM_EPOCHS}_${LATENT_DIM}_${BATCH_SIZE}_env${ENV_ID}"
SAVE_DIR="${central_results_folder}/${SAVE_NAME}"
mkdir -p "$SAVE_DIR"

echo "Using GPU ID: $CUDA_VISIBLE_DEVICES" > "${SAVE_DIR}/gpu_used.txt"

python3 vae_main.py \
  --dataset concon \
  --env_id $ENV_ID \
  --beta 40.0 \
  --tcvae \
  --beta-anneal \
  --lambda-anneal \
  --mss \
  --conv \
  --save "$SAVE_DIR" \
  > "${SAVE_DIR}/output_env${ENV_ID}.txt"
