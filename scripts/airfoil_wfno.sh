export CUDA_VISIBLE_DEVICES=1

python Latent-Spectral-Models-main/exp_airfoils.py \
  --data-path /data/petrol/Occulmens/PDEBench-main/pdebench/save/airfoils/ \
  --ntrain 1000 \
  --ntest 100 \
  --ntotal 1100 \
  --in_dim 2 \
  --out_dim 1 \
  --h 221 \
  --w 51 \
  --h-down 1 \
  --w-down 1 \
  --batch-size 20 \
  --learning-rate 0.001 \
  --model WFNO_2D \
  --d-model 64 \
  --num-basis 12 \
  --num-token 4 \
  --patch-size 14,4 \
  --padding 13,3 \
  --model-save-path ./checkpoints/airfoil \
  --model-save-name lsm.pt