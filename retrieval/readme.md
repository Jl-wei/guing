Train on all mix data
```bash
python train_clip.py \
    --output_dir ./models/clip-base-mix-all \
    --model_name_or_path "openai/clip-vit-base-patch32" \
    --dataset_name ../dataset/mix-caption/all-dataset.py \
    --image_column image \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train \
    --max_seq_length="77" \
    --per_device_train_batch_size="128" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --num_train_epochs=5 \
    --overwrite_output_dir \
    --save_strategy epoch \
    --seed 916
```

Train on training set of all mix data
```bash
python train_clip.py \
    --output_dir ./models/clip-base-mix-all-train \
    --model_name_or_path "openai/clip-vit-base-patch32" \
    --dataset_name ../dataset/mix-caption/train-all-dataset.py \
    --image_column image \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train \
    --max_seq_length="77" \
    --per_device_train_batch_size="128" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --num_train_epochs=5 \
    --overwrite_output_dir \
    --save_strategy epoch \
    --seed 916
```


Train on mix of Screen2Words and Clarity dataset
```bash
python train_clip.py \
    --output_dir ./models/clip-base-mix-s2w-cla \
    --model_name_or_path "openai/clip-vit-base-patch32" \
    --dataset_name ../dataset/mix-caption/s2w-cla-dataset.py \
    --image_column image \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train \
    --max_seq_length="77" \
    --per_device_train_batch_size="128" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --num_train_epochs=5 \
    --overwrite_output_dir \
    --save_strategy epoch \
    --seed 916
```

Train on training set of the mix of Screen2Words and Clarity dataset
```bash
python train_clip.py \
    --output_dir ./models/clip-base-mix-s2w-cla-train \
    --model_name_or_path "openai/clip-vit-base-patch32" \
    --dataset_name ../dataset/mix-caption/train-s2w-cla-dataset.py \
    --image_column image \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train \
    --max_seq_length="77" \
    --per_device_train_batch_size="128" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --num_train_epochs=5 \
    --overwrite_output_dir \
    --save_strategy epoch \
    --seed 916
```

