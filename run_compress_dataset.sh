# wikipedia-cn-20230720-documents_all.json can be found at https://huggingface.co/datasets/howard-hou/WikiQA-LongForm/blob/main/wikipedia-cn-20230720-documents_all.json


python scripts/compress_dataset.py \
    howard-hou/WikiQA-LongForm-subset-rerank \
    howard-hou/IACC-compressor-small \
    wikipedia-cn-20230720-documents_all.json \
    output_path \
    --device cuda --keep_ratio 0.65 --max_new_tokens 16
