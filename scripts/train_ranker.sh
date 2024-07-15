CUDA_VISIBLE_DEVICES=0 python train_ranker.py  \
	--model_name_or_path=/data/huggingface_model/umt5-small \
	--dataset_path=data/final_dataset/large_ranker_dataset \
	--model_output_dir=saved_model/small-ranker_large-dataset \
	--batch_size=4 \
	--num_epochs=1
