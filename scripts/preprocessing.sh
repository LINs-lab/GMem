export PYTHONPATH=./

python extract_features.py \
    --data_path  data/ImageNet-1K/train\
    --data_split imagenet_train \
    --output_path data/preprocessed/in1k256 \
    --config tokenizer/configs/vavae_f16d32.yaml \
