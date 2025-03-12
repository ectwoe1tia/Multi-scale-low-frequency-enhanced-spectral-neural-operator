export CUDA_VISIBLE_DEVICES=1

python Latent-Spectral-Models-main/exp_darcy.py \
  --data-path /data/petrol/Occulmens/PDEBench-main/pdebench/save/Darcy/ \
  --ntrain 1000 \
  --ntest 200 \
  --ntotal 1200 \
  --in_dim 1 \
  --out_dim 1 \
  --h 421 \
  --w 421 \
  --h-down 5 \
  --w-down 5 \
  --batch-size 20 \
  --learning-rate 0.001 \
  --model FNO_2D \
  --d-model 16 \
  --num-basis 12 \
  --num-token 4 \
  --patch-size 6,6 \
  --padding 11,11 \
  --model-save-path ./data/petrol/Occulmens/PDEBench-main/pdebench/save/Darcy/ \
  --model-save-name fno_Darcy.pt
