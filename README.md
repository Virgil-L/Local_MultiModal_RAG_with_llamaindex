# MultiModal RAG with LlamaIndex + Qdrant + Local Vision-LLM & Embedding models

## Overview

This project is implemented within the framework of LlamaIndex, using multiple custom components to achieve a fully localized multimodal document-QA system without relying on any APIs or remote resources.

Below are the main tools/models used:

- **PDF parser**: [SciPDF Parser](https://github.com/titipata/scipdf_parser)

- **RAG Framework**: [LlamaIndex](https://github.com/run-llama/llama_index)

- **Vector DataBase**: [Qdrant](https://qdrant.tech/)

- **Vison-LLM**: A [gguf quantized 7B version](https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf) of [LLaVA](https://llava-vl.github.io/)

- **LLM Inference Framework**: [llama.cpp](https://github.com/ggerganov/llama.cpp) & [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)

- **Embedding Models**:
    - **BGE** models for text [embedding](https://huggingface.co/BAAI/bge-small-en-v1.5) and [reranking](https://huggingface.co/BAAI/bge-reranker-base)
    
    - Efficient **SPLADE** models ([doc](https://huggingface.co/naver/efficient-splade-VI-BT-large-doc), [query](https://huggingface.co/naver/efficient-splade-VI-BT-large-query)) for sparse retrieval 
    

    - [**CLIP**](https://huggingface.co/openai/clip-vit-base-patch32) for query-to-image retrieval


## Environment

WSL2 on Windows 10, Ubuntu 22.04 

CUDA Toolkit Version 12.3


## Library Installation

**SciPDF Parser**

Follow the steps from https://github.com/titipata/scipdf_parser/blob/master/README.md. 

Install this branch to avoid exceeding the request time limit when dealing with large PDF files.
```
pip install https://github.com/Virgil-L/scipdf_parser.git
```
To save memory and computing cost, this project uses a lightweight image  of [GROBID](https://github.com/kermitt2/grobid), run `serve_grobid_light.sh` to get it.

**LlamaIndex**

```
pip install -q llama-index llama-index-embeddings-huggingface
```

**Qdrant**
```
pip install qdrant-client
```

**llama-cpp-python**

```
# Linux and Mac
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

## Examples

Download the sample pdf data
```
wget --user-agent "Mozilla" "https://arxiv.org/pdf/2307.09288.pdf" -O "data/paper_PDF/llama2.pdf"
wget --user-agent "Mozilla" "https://arxiv.org/pdf/2310.03744.pdf" -O "data/paper_PDF/llava-vl.pdf"

wget --user-agent "Mozilla" "https://arxiv.org/pdf/1706.03762.pdf" -O "data/paper_PDF/attention.pdf"
wget --user-agent "Mozilla" "https://arxiv.org/pdf/1810.04805.pdf" -O "data/paper_PDF/bert.pdf"
```

Parse pdf files to extract text and image elemnts
```
python parse_pdf.py \
    --pdf_folder "./data/paper_PDF" \
    --image_resolution 300 \
    --max_timeout 120
```

Create Qdrant collections, build and ingest text and image nodes
```
python build_qdrant_collections.py \
    --pdf_folder "./data/paper_PDF" \
    --storage_path "./qdrant_db" \
    --text_embedding_model "./embedding_models/bge-small-en-v1.5" \
    --sparse_text_embedding_model "./embedding_models/efficient-splade-VI-BT-large-doc" \
    --image_embedding_model "./embedding_models/clip-vit-base-patch32" \
    --chunk_size 384 \
    --chunk_overlap 32
```