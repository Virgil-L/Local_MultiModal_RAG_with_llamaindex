from flask import Flask, request, jsonify
import os
from functools import partial
import yaml
import json
import warnings
warnings.filterwarnings(action="ignore", category=FutureWarning)

import numpy as np
from llama_index.legacy.schema import QueryBundle


class JSON_Improved(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return int(obj)
        elif isinstance(obj, np.float16):
            return float(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        else:
            return super(JSON_Improved, self).default(obj)


from flask.json.provider import JSONProvider

class CustomJSONProvider(JSONProvider):
    
    def dumps(self, obj, **kwargs):
        return json.dumps(obj, **kwargs, cls=JSON_Improved)
    
    def loads(self, s: str | bytes, **kwargs):
        return json.loads(s, **kwargs)


Flask.json_provider_class = CustomJSONProvider
app = Flask(__name__)


with open("retriever_config.yaml", 'r') as f:
    try:
        config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)


mm_retriever = None
client = None

def initialize_service(config):
    global mm_retriever, client
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qd_models
    from src.custom_vectore_store import MultiModalQdrantVectorStore
    from src.custom_embeddings import custom_sparse_doc_vectors, custom_sparse_query_vectors
    from llama_index.legacy.vector_stores import QdrantVectorStore

    try:
        if os.path.exists(os.path.join(config['qdrant_path'], ".lock")):
            os.remove(os.path.join(config['qdrant_path'], ".lock"))
        client = QdrantClient(path=config['qdrant_path'])
        print("Connected to Qdrant")
    except Exception as e:
        print("Error connecting to Qdrant: ", str(e))
        
    # load model
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    splade_q_tokenizer = AutoTokenizer.from_pretrained(config['splade_query_path'], clean_up_tokenization_spaces=True)
    splade_q_model = AutoModelForMaskedLM.from_pretrained(config['splade_query_path'])

    splade_d_tokenizer = AutoTokenizer.from_pretrained(config['splade_doc_path'], clean_up_tokenization_spaces=True)
    splade_d_model = AutoModelForMaskedLM.from_pretrained(config['splade_doc_path'])

    custom_sparse_doc_fn = partial(custom_sparse_doc_vectors, splade_d_tokenizer, splade_d_model, 512)
    custom_sparse_query_fn = partial(custom_sparse_query_vectors, splade_q_tokenizer, splade_q_model, 512)

    text_store = QdrantVectorStore(
        client=client,
        collection_name=config['text_collection_name'],
        enable_hybrid=True,
        sparse_query_fn=custom_sparse_query_fn,
        sparse_doc_fn=custom_sparse_doc_fn,
        stores_text=True,
    )

    image_store = MultiModalQdrantVectorStore(
        client=client,
        collection_name=config['image_collection_name'],
        enable_hybrid=True,
        sparse_query_fn=custom_sparse_query_fn,
        sparse_doc_fn=custom_sparse_doc_fn,
        stores_text=False,
    )


    from llama_index.legacy.embeddings import HuggingFaceEmbedding
    from src.custom_embeddings import CustomizedCLIPEmbedding


    text_embedding = HuggingFaceEmbedding(model_name=config['embedding_path'], device="cpu", pooling="mean")
    image_embedding = CustomizedCLIPEmbedding(model_name=config['image_encoder_path'], device="cpu")
    
    from llama_index.core.postprocessor import SentenceTransformerRerank

    reranker = SentenceTransformerRerank(
            model=config['reranker_path'],
            top_n=3,
            device="cpu",
            keep_retrieval_score=False,
        )
    from src.mm_retriever import MultiModalQdrantRetriever

    mm_retriever = MultiModalQdrantRetriever(
        text_vector_store = text_store,
        image_vector_store = image_store, 
        text_embed_model = text_embedding, 
        mm_embed_model = image_embedding,
        reranker = reranker,
        text_similarity_top_k = config['text_similarity_top_k'],
        text_sparse_top_k = config['text_sparse_top_k'],
        text_rerank_top_n = config['text_rerank_top_n'],
        image_similarity_top_k = config['image_similarity_top_k'],
        image_sparse_top_k = config['image_sparse_top_k'],
        image_rerank_top_n = config['image_rerank_top_n'],
        sparse_query_fn = custom_sparse_query_fn,
    )


def initialize():
    try:
        initialize_service(config=config)
        print("########## Retriever Service initialized. ##########")
        return jsonify({"status": "success", "message": "Service initialized."}), 200
    except Exception as e:
        print(f"Error initializing service: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

with app.app_context():
    initialize()


def process_query(query, text_topk=None, image_topk=None):

    query_bundle=QueryBundle(query_str=query)

    text_query_result = mm_retriever.retrieve_text_nodes(query_bundle=query_bundle, query_mode="hybrid")
    print("Retrieved text nodes")
    reranked_text_nodes = mm_retriever.rerank_text_nodes(query_bundle, text_query_result, text_rerank_top_n=text_topk)
    print("Reranked text nodes")
    image_query_result = mm_retriever.retrieve_image_nodes(query_bundle=query_bundle, query_mode="hybrid")
    print("Retrieved image nodes")
    reranked_image_nodes = mm_retriever.rerank_image_nodes(query_bundle, image_query_result, image_rerank_top_n=image_topk)
    print("Reranked image nodes")
    
    # for item in reranked_image_nodes:
    #     item.node.metadata['vectors'] = None
    # for item in reranked_text_nodes:
    #     item.node.metadata['vectors'] = None
    #     del item.node.metadata['regionBoundary']
    #     del item.node.metadata['captionBoundary']
    return reranked_text_nodes, reranked_image_nodes


@app.route("/api", methods=['POST'])
def handle_request():
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({"status": "error", "message": "Invalid request."}), 400
        
        text_topk = data.get('text_topk', None)
        image_topk = data.get('image_topk', None)
        query = data['query']
        # 处理查询
        text_nodes, image_nodes = process_query(query, text_topk=text_topk, image_topk=image_topk)
        text_nodes = [node.to_dict() for node in text_nodes]
        image_nodes = [node.to_dict() for node in image_nodes]
        return jsonify({"status": "success", 
                        "query": query, 
                        "text_result": text_nodes, 
                        "image_result": image_nodes}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="127.0.0.1", port=5000)