from llama_index.legacy.schema import QueryBundle
from llama_index.legacy.retrievers import BaseRetriever
from llama_index.legacy.schema import NodeWithScore
from llama_index.legacy.vector_stores import VectorStoreQuery
from llama_index.legacy.vector_stores.types import VectorStoreQueryMode
from typing import Any, List, Optional


class MultiModalQdrantRetriever(BaseRetriever):
    """Retriever over a qdrant vector store."""

    def __init__(
        self,
        text_vector_store: QdrantVectorStore = None,
        image_vector_store: MultiModalQdrantVectorStore = None,
        
        text_embed_model: Any = None,
        mm_embed_model: Any = None,


        text_similarity_top_k: int = 5,
        text_sparse_top_k: int = 5,
        image_similarity_top_k: int = 5,
        image_sparse_top_k: int = 5,
    ) -> None:
        """Init params."""

        self._text_vector_store = text_vector_store
        self._image_vector_store = image_vector_store
        self._text_embed_model = text_embed_model
        self._mm_embed_model = mm_embed_model

        self._text_similarity_top_k = text_similarity_top_k
        self._text_sparse_top_k = text_sparse_top_k
        self._image_similarity_top_k = image_similarity_top_k
        self._image_sparse_top_k = image_sparse_top_k

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

    def _retrieve(self, query_bundle: QueryBundle, query_mode: str="hybrid", metadata_filters=None):

        """ Deprecated abstract retrieve method from the BaseRetriever, this can only retrieve text nodes."""

        ###TODO: rewrite this method to use the new retrieve_text_nodes and retrieve_image_nodes
        # return {
        #     "text_nodes": self.retrieve_text_nodes(query_bundle, query_mode, metadata_filters),
        #     "image_nodes": self.retrieve_image_nodes(query_bundle, query_mode, metadata_filters)
        # }

        query_embedding = self._text_embed_model.get_query_embedding(
            query_bundle.query_str
        )        
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._text_similarity_top_k,
            mode="default",
        )

        query_result = self._text_vector_store.query(vector_store_query)
        
        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

    ### TODO: rewrite the following methods to use the new retrieve_text_nodes and retrieve_image_nodes

    # def retrieve(self, str_or_query_bundle: QueryType):
    #     return

    # async def _aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    #     """Asynchronously retrieve nodes given query.

    #     Implemented by the user.

    #     """
    #     return self._retrieve(query_bundle)

    # async def aretrieve(self, str_or_query_bundle: QueryType):
    #     return


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

    image_query_result = mm_retriever.retrieve_image_nodes(query_bundle=QueryBundle(query_str="How does Llama 2 perform compared to other open-source models?"), query_mode="hybrid")

    print(text_query_result)
    print(image_query_result)