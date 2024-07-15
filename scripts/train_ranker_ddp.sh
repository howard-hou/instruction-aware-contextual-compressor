CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --standalone --nproc_per_node=4 train_ranker.py  \
        --model_name_or_path=/data/huggingface_model/google/umt5-base \
        --dataset_path=data/final_dataset/large_ranker_dataset \
        --model_output_dir=saved_model/base-ranker_large-dataset \
        --gradient_accumulation_steps=32 \
        --batch_size=1 \
        --num_epochs=1
