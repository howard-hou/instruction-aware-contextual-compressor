# Instruction-Aware Contextual Compressor

Welcome to the Instruction-Aware Contextual Compressor repository! This project is designed to compress context for RAG in LLM.

## Key Features

- **Contextual Compression**: Intelligently compresses data based on the context it is used in.
- **Instruction-Aware**: Adapts compression techniques based on the instructions provided.

## Getting Started

To get started with the Instruction-Aware Contextual Compressor, follow these simple steps:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/howard-hou/instruction-aware-contextual-compressor.git
   ```

2. **Install Dependencies**
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirement.txt
   ```

## Reproducing Experimental Results

- 1. Ensure to download the proper checkpoint: howard-hou/IACC-compressor-small
- 2. Acquire the appropriate dataset via the provided [link](https://huggingface.co/datasets/howard-hou/WikiQA-LongForm-subset-rerank)
- 3  Acquire the appropriate document collection via the provided [line](https://huggingface.co/datasets/howard-hou/WikiQA-LongForm/blob/main/wikipedia-cn-20230720-documents_all.json)
- 4. Run the following command to reproduce the results:
```bash run_compress_dataset.sh```

## Checkpoints

weights and checkpoints are stored in the huggingface model hub.

- [ranker](https://huggingface.co/howard-hou/IACC-ranker-small)
- [compressor](https://huggingface.co/howard-hou/IACC-compressor-small)

## Citation

If you find this project useful, please consider citing it:

```bibtex
@article{howard-hou2022instruction-aware-compressor,
  title={Instruction-Aware Contextual Compressor},
  author={Howard Hou},
  journal={GitHub},
  volume={1},
  number={1},
  pages={1-10},
  year={2022},
  publisher={GitHub}
}
```

## License

This project is licensed under the [Apache-2.0 License](LICENSE).