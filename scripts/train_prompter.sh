CUDA_VISIBLE_DEVICES=1 python train_prompter.py  \
	--model_name_or_path=/data/huggingface_model/RankingPrompterForPreTraining-small \
	--dataset_path=data/final_dataset/prompter_dataset \
	--model_output_dir=saved_model/small-prompter2_with_pretrain \
	--batch_size=4 \
	--num_epochs=1
