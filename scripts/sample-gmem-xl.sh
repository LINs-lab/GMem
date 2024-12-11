export PYTHONPATH=.

torchrun --nnodes=1 --nproc_per_node=8 engine/generate.py \
  --model SiT-XL/2 \
  --num-fid-samples 50000 \
  --path-type=linear \
  --encoder-depth=8 \
  --projector-embed-dims=768 \
  --per-proc-batch-size=64 \
  --mode=sde \
  --num-steps=250 \
  --cfg-scale=0 \
  --guidance-high=0 \
  --use-feature-condition \
  --ckpt PATH/TO/CKPT \
  --bank-path PATH/TO/BANK \
