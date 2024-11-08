from llama_index.legacy.schema import QueryBundle
from llama_index.legacy.retrievers import BaseRetriever
from llama_index.legacy.schema import NodeWithScore
from llama_index.legacy.vector_stores import VectorStoreQuery
from llama_index.legacy.vector_stores.types import VectorStoreQueryMode
from llama_index.legacy.vector_stores.qdrant import QdrantVectorStore
from llama_index.legacy.schema import QueryType
from .custom_vectore_store import MultiModalQdrantVectorStore
from typing import Any, List, Optional

import numpy as np
from qdrant_client.http import models as rest

def compute_cosine_similarity(a, b):
    if isinstance(a, rest.SparseVector):
        intersect_indices = set(a.indices) & set(b.indices)
        if len(intersect_indices) == 0:
            return 0
        else:
            a_intersect_values = [a.values[a.indices.index(i)] for i in intersect_indices]
            b_intersect_values = [b.values[b.indices.index(i)] for i in intersect_indices]
            
        
            return np.dot(a_intersect_values, b_intersect_values) / (np.linalg.norm(a.values) * np.linalg.norm(b.values))
    else:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



class MultiModalQdrantRetriever(BaseRetriever):
    """Retriever over a qdrant vector store."""

    def __init__(
        self,
        text_vector_store: QdrantVectorStore,
        image_vector_store: MultiModalQdrantVectorStore,
        
        text_embed_model: Any,
        mm_embed_model: Any,
        reranker: Optional[Any]=None,


        text_similarity_top_k: int = 5,
        text_sparse_top_k: int = 5,
        text_rerank_top_n: int = 3,
        image_similarity_top_k: int = 5,
        image_sparse_top_k: int = 5,
        image_rerank_top_n: int = 1,

        sparse_query_fn: Optional[Any] = None,
    ) -> None:
        """Init params."""

        self._text_vector_store = text_vector_store
        self._image_vector_store = image_vector_store
        self._text_embed_model = text_embed_model
        self._mm_embed_model = mm_embed_model
        self._reranker = reranker

        self._text_similarity_top_k = text_similarity_top_k
        self._text_sparse_top_k = text_sparse_top_k
        self._text_rerank_top_n = text_rerank_top_n
        self._image_similarity_top_k = image_similarity_top_k
        self._image_sparse_top_k = image_sparse_top_k
        self._image_rerank_top_n = image_rerank_top_n

        self._sparse_query_fn = sparse_query_fn

        super().__init__()

    def retrieve_text_nodes(self, query_bundle: QueryBundle, query_mode: str="hybrid", metadata_filters=None):

        query_embedding = self._text_embed_model.get_query_embedding(
            query_bundle.query_str
        )

        # query with dense text embedding
        dense_query = VectorStoreQuery(
            query_str=query_bundle.query_str,
            query_embedding=query_embedding,
            similarity_top_k=self._text_similarity_top_k,
            sparse_top_k=self._text_sparse_top_k,
            mode=VectorStoreQueryMode.DEFAULT,
            filters=metadata_filters,
        )

        # query with sparse text vector
        sparse_query = VectorStoreQuery(
            query_str=query_bundle.query_str,
            query_embedding=query_embedding,
            similarity_top_k=self._text_similarity_top_k,
            sparse_top_k=self._text_sparse_top_k,
            mode=VectorStoreQueryMode.SPARSE,
            filters=metadata_filters,
        )

        # mm_query = VectorStoreQuery(...)
        
        # returns a VectorStoreQueryResult
        if query_mode == "default":
            dense_query_result = self._text_vector_store.query(dense_query)
            
            return {
                "text-dense": dense_query_result
            }

        elif query_mode == "sparse":
            sparse_query_result = self._text_vector_store.query(sparse_query)
            
            return {
                "text-sparse": sparse_query_result
            }


        elif query_mode == "hybrid":
            dense_query_result = self._text_vector_store.query(dense_query)
            sparse_query_result = self._text_vector_store.query(sparse_query)

            return {
                "text-dense": dense_query_result,
                "text-sparse": sparse_query_result
            }

        else:
            raise ValueError(f"Invalid text-to-text query mode: {query_mode}, must be one of ['default', 'sparse', 'hybrid']")


    def rerank_text_nodes(self, query_bundle: QueryBundle, text_retrieval_result, text_rerank_top_n = None):

        text_node_ids, text_nodes = [], []
        # text_node_ids, text_node, text_node_scores = [], [], []

        for key in text_retrieval_result.keys():
            text_node_ids += text_retrieval_result[key].ids
            text_nodes += text_retrieval_result[key].nodes
            # text_node_scores += text_retrieval_result[key].similarities

        if text_rerank_top_n is None:
            text_rerank_top_n = self._text_rerank_top_n
        else:
            text_rerank_top_n = min(text_rerank_top_n, len(text_nodes))
        self._reranker.top_n = text_rerank_top_n

        # drop duplicate nodes from sparse retrival and dense retrival        
        unique_node_indices = list(set([text_node_ids.index(x) for x in text_node_ids if text_node_ids.count(x) >= 1]))

        ## reserve similarity score of retrival stage
        # text_nodes = [text_nodes[i] for i in unique_node_indices]
        # text_node_scores = [text_node_scores[i] for i in unique_node_indices]
        # text_nodes_with_score = [NodeWithScore(node=_[0], score=_[1]) for _ in list(zip(text_nodes, text_node_scores))]

        # set similarity score to 0.0 only for format consistency in reranking stage
        text_nodes_with_score = [NodeWithScore(node=text_nodes[i], score=0.0) for i in unique_node_indices] 
        

        return self._reranker._postprocess_nodes(nodes=text_nodes_with_score, query_bundle=query_bundle)


    def retrieve_image_nodes(self, query_bundle: QueryBundle, query_mode: str="default", metadata_filters=None):

        
        if query_mode == "default": # Default: query with dense multi-modal embedding only
            mm_query = VectorStoreQuery(
                query_str=query_bundle.query_str,
                query_embedding=self._mm_embed_model.get_query_embedding(query_bundle.query_str),
                similarity_top_k=self._image_similarity_top_k,
                mode=VectorStoreQueryMode.DEFAULT,
                filters=metadata_filters,
            )
            mm_query_result = self._image_vector_store.text_to_image_query(mm_query)
            
            return {
                "multi-modal": mm_query_result
            }


        elif query_mode == "text-dense":
            text_dense_query = VectorStoreQuery(
                query_str=query_bundle.query_str,
                query_embedding=self._text_embed_model.get_query_embedding(query_bundle.query_str),
                similarity_top_k=self._image_similarity_top_k,
                mode=VectorStoreQueryMode.DEFAULT,
                filters=metadata_filters,
            )
            text_dense_query_result = self._image_vector_store.text_to_caption_query(text_dense_query)
            
            return {
                "text-dense": text_dense_query_result
            }


        elif query_mode == "text-sparse":
            text_sparse_query = VectorStoreQuery(
                query_str=query_bundle.query_str,
                #query_embedding=self._text_embed_model.get_query_embedding(query_bundle.query_str),
                similarity_top_k=self._image_sparse_top_k,
                mode=VectorStoreQueryMode.SPARSE,
                filters=metadata_filters,
            )
            text_sparse_query_result = self._image_vector_store.text_to_caption_query(text_sparse_query)
            
            return {
                "text-sparse": text_sparse_query_result
            }

        elif query_mode == "hybrid":
            mm_query = VectorStoreQuery(
                query_str=query_bundle.query_str,
                query_embedding=self._mm_embed_model.get_query_embedding(query_bundle.query_str),
                similarity_top_k=self._image_similarity_top_k,
                mode=VectorStoreQueryMode.DEFAULT,
                filters=metadata_filters,
            )
            mm_query_result = self._image_vector_store.text_to_image_query(mm_query)

            text_dense_query = VectorStoreQuery(
                query_str=query_bundle.query_str,
                query_embedding=self._text_embed_model.get_query_embedding(query_bundle.query_str),
                similarity_top_k=self._image_similarity_top_k,
                mode=VectorStoreQueryMode.DEFAULT,
                filters=metadata_filters,
            )
            text_dense_query_result = self._image_vector_store.text_to_caption_query(text_dense_query)
            
            text_sparse_query = VectorStoreQuery(
                query_str=query_bundle.query_str,
                #query_embedding=self._text_embed_model.get_query_embedding(query_bundle.query_str),
                similarity_top_k=self._image_sparse_top_k,
                mode=VectorStoreQueryMode.SPARSE,
                filters=metadata_filters,
            )
            text_sparse_query_result = self._image_vector_store.text_to_caption_query(text_sparse_query)

            return {
                "multi-modal": mm_query_result,
                "text-dense": text_dense_query_result,
                "text-sparse": text_sparse_query_result
            }

        else:
            raise ValueError(f"Invalid text-to-image query mode: {query_mode}, must be one of ['default', 'text-dense', 'text-sparse', 'hybrid']")



    def rerank_image_nodes(self, query_bundle: QueryBundle, image_retrieval_result, image_rerank_top_n = None):
       
        image_nodes, image_node_ids =  [], []
        for key in image_retrieval_result.keys():
            image_node_ids += image_retrieval_result[key].ids
            image_nodes += image_retrieval_result[key].nodes

            # image_similarities = np.array(image_retrieval_result[key].similarities)
            # normed_similarities = (image_similarities - image_similarities.mean()) / image_similarities.std()
            # image_node_scores += normed_similarities.tolist()

        unique_node_indices = list(set([image_node_ids.index(x) for x in image_node_ids if image_node_ids.count(x) >= 1]))
        image_node_nodes = [image_nodes[i] for i in unique_node_indices]
        
        if image_rerank_top_n is None:
            image_rerank_top_n = self._image_rerank_top_n
        else:
            image_rerank_top_n = min(image_rerank_top_n, len(image_node_nodes))
                                     
        query_str = query_bundle.query_str
        similarity_scores = {key: [] for key in image_retrieval_result.keys()}

        for key in image_retrieval_result.keys():
            if key == "text-dense":
                query_embedding = self._text_embed_model.get_query_embedding(query_str)
                for i, node in enumerate(image_node_nodes):
                    node_embedding = node.metadata['vectors'][key]
                    similarity_scores[key].append(compute_cosine_similarity(query_embedding, node_embedding))

            elif key == "text-sparse":
                query_embedding = self._sparse_query_fn(query_str)
                query_embedding = rest.SparseVector(indices=query_embedding[0][0], values=query_embedding[1][0])
                for i, node in enumerate(image_node_nodes):
                    node_embedding = node.metadata['vectors'][key]
                    similarity_scores[key].append(compute_cosine_similarity(query_embedding, node_embedding))

            elif key == "multi-modal":
                query_embedding = self._mm_embed_model.get_query_embedding(query_str)
                for i, node in enumerate(image_node_nodes):
                    node_embedding = node.metadata['vectors'][key]
                    similarity_scores[key].append(compute_cosine_similarity(query_embedding, node_embedding))


        rerank_scores = np.zeros(len(image_node_nodes))
        for key in similarity_scores.keys():
            similarity_scores[key] = np.array(similarity_scores[key])
            similarity_scores[key] = (similarity_scores[key] - similarity_scores[key].mean()) / similarity_scores[key].std()
            rerank_scores += similarity_scores[key]

        rerank_score_with_index = list(zip(rerank_scores, range(len(image_node_nodes))))
        rerank_score_with_index = sorted(rerank_score_with_index, key=lambda x: x[0], reverse=True)
        topn_image_nodes = [NodeWithScore(node=image_node_nodes[_[1]], score=_[0]) for _ in rerank_score_with_index][:image_rerank_top_n]

        for node in topn_image_nodes:
            node.node.metadata['vectors'] = None

        return topn_image_nodes






    ### TODO: rewrite the following methods to use the new retrieve_text_nodes and retrieve_image_nodes
    
    def _retrieve(self, query_bundle: QueryBundle, query_mode: str="hybrid", metadata_filters=None):

        """ Deprecated abstract retrieve method from the BaseRetriever, this can only retrieve text nodes."""

        raise NotImplementedError("This method is deprecated, please use retrieve_text_nodes and retrieve_image_nodes instead.")

        ###TODO: rewrite this method to use the new retrieve_text_nodes and retrieve_image_nodes
        # return {
        #     "text_nodes": self.retrieve_text_nodes(query_bundle, query_mode, metadata_filters),
        #     "image_nodes": self.retrieve_image_nodes(query_bundle, query_mode, metadata_filters)
        # }
    def retrieve(self, str_or_query_bundle: QueryType):
        raise NotImplementedError("This method is deprecated, please use retrieve_text_nodes and retrieve_image_nodes instead.")

    async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Asynchronously retrieve nodes given query.

        Implemented by the user.

        """
        return self._retrieve(query_bundle)

    async def aretrieve(self, str_or_query_bundle: QueryType):
        return


if __name__ == "__main__":

    from llama_index.legacy.vector_stores import QdrantVectorStore
    from custom_vectore_store import MultiModalQdrantVectorStore
    from custom_embeddings import custom_sparse_doc_vectors, custom_sparse_query_vectors

    from functools import partial

    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qd_models

    try:
        client = QdrantClient(path="qdrant_db")
    except:
        print("Qdrant server not running, please start the server and try again.")
        pass

    # client = QdrantClient(path="qdrant_db")


    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM


    from llama_index.legacy.embeddings import HuggingFaceEmbedding
    from custom_embeddings import CustomizedCLIPEmbedding

    BGE_PATH = "./embedding_models/bge-small-en-v1.5"
    CLIP_PATH = "./embedding_models/clip-vit-base-patch32"
    bge_embedding = HuggingFaceEmbedding(model_name=BGE_PATH, device="cpu", pooling="mean")
    clip_embedding = CustomizedCLIPEmbedding(model_name=CLIP_PATH, device="cpu")

    SPLADE_QUERY_PATH = "./embedding_models/efficient-splade-VI-BT-large-query"
    splade_q_tokenizer = AutoTokenizer.from_pretrained(SPLADE_QUERY_PATH)
    splade_q_model = AutoModelForMaskedLM.from_pretrained(SPLADE_QUERY_PATH)

    SPLADE_DOC_PATH = "./embedding_models/efficient-splade-VI-BT-large-doc"
    splade_d_tokenizer = AutoTokenizer.from_pretrained(SPLADE_DOC_PATH)
    splade_d_model = AutoModelForMaskedLM.from_pretrained(SPLADE_DOC_PATH)

    custom_sparse_doc_fn = partial(custom_sparse_doc_vectors, splade_d_tokenizer, splade_d_model, 512)
    custom_sparse_query_fn = partial(custom_sparse_query_vectors, splade_q_tokenizer, splade_q_model, 512)

    text_store = QdrantVectorStore(
        client=client,
        collection_name="text_collection",
        enable_hybrid=True,
        sparse_query_fn=custom_sparse_query_fn,
        sparse_doc_fn=custom_sparse_doc_fn,
        stores_text=True,
    )

    image_store = MultiModalQdrantVectorStore(
        client=client,
        collection_name="image_collection",
        enable_hybrid=True,
        sparse_query_fn=custom_sparse_query_fn,
        sparse_doc_fn=custom_sparse_doc_fn,
        stores_text=False,
    )


    mm_retriever = MultiModalQdrantRetriever(
        text_vector_store = text_store,
        image_vector_store = image_store, 
        text_embed_model = bge_embedding, 
        mm_embed_model = clip_embedding,
    )

    text_query_result = mm_retriever.retrieve_text_nodes(query_bundle=QueryBundle(query_str="How does Llama 2 perform compared to other open-source models?"), query_mode="hybrid")

    image_retrieval_result = mm_retriever.retrieve_image_nodes(query_bundle=QueryBundle(query_str="How does Llama 2 perform compared to other open-source models?"), query_mode="hybrid")

    print(text_query_result)
    print(image_retrieval_result)