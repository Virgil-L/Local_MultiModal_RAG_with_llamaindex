"""
Multi-modal Qdrant vector store index.

An index that is built on top of an existing Qdrant collection for text-image retrieval.

"""



import logging
from typing import Any, List, Optional, Tuple, cast

import qdrant_client
from grpc import RpcError

from llama_index.legacy.vector_stores.qdrant import QdrantVectorStore
from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.schema import BaseNode, MetadataMode, TextNode, ImageNode
from llama_index.legacy.utils import iter_batch
from llama_index.legacy.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.legacy.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
    node_to_metadata_dict,
)
from llama_index.legacy.vector_stores.qdrant_utils import (
    HybridFusionCallable,
    SparseEncoderCallable,
    default_sparse_encoder,
    relative_score_fusion,
)


from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchExcept,
    MatchText,
    MatchValue,
    Payload,
    Range,
)

logger = logging.getLogger(__name__)
import_err_msg = (
    "`qdrant-client` package not found, please run `pip install qdrant-client`"
)


class MultiModalQdrantVectorStore(QdrantVectorStore):
    """
    Multi-modal Qdrant Vector Store.

    In this vector store, embeddings and docs are stored within a
    Qdrant collection.

    During query time, the index uses Qdrant to query for the top
    k most similar nodes.

    Args:
        collection_name: (str): name of the Qdrant collection
        client (Optional[Any]): QdrantClient instance from `qdrant-client` package
        aclient (Optional[Any]): AsyncQdrantClient instance from `qdrant-client` package
        url (Optional[str]): url of the Qdrant instance
        api_key (Optional[str]): API key for authenticating with Qdrant
        batch_size (int): number of points to upload in a single request to Qdrant. Defaults to 64
        parallel (int): number of parallel processes to use during upload. Defaults to 1
        max_retries (int): maximum number of retries in case of a failure. Defaults to 3
        client_kwargs (Optional[dict]): additional kwargs for QdrantClient and AsyncQdrantClient
        enable_hybrid (bool): whether to enable hybrid search using dense and sparse vectors
        sparse_doc_fn (Optional[SparseEncoderCallable]): function to encode sparse vectors
        sparse_query_fn (Optional[SparseEncoderCallable]): function to encode sparse queries
        hybrid_fusion_fn (Optional[HybridFusionCallable]): function to fuse hybrid search results
    """

    stores_text: bool = True
    flat_metadata: bool = False

    collection_name: str
    path: Optional[str]
    url: Optional[str]
    api_key: Optional[str]
    batch_size: int
    parallel: int
    max_retries: int
    client_kwargs: dict = Field(default_factory=dict)
    enable_hybrid: bool

    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()
    _collection_initialized: bool = PrivateAttr()
    _sparse_doc_fn: Optional[SparseEncoderCallable] = PrivateAttr()
    _sparse_query_fn: Optional[SparseEncoderCallable] = PrivateAttr()
    _hybrid_fusion_fn: Optional[HybridFusionCallable] = PrivateAttr()

    def __init__(
        self,
        collection_name: str,
        client: Optional[Any] = None,
        aclient: Optional[Any] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 64,
        parallel: int = 1,
        max_retries: int = 3,
        client_kwargs: Optional[dict] = None,
        enable_hybrid: bool = False,
        sparse_doc_fn: Optional[SparseEncoderCallable] = None,
        sparse_query_fn: Optional[SparseEncoderCallable] = None,
        hybrid_fusion_fn: Optional[HybridFusionCallable] = None,
        **kwargs: Any,
    ) -> None:
        """Init params."""
        if (
            client is None
            and aclient is None
            and (url is None or api_key is None or collection_name is None)
        ):
            raise ValueError(
                "Must provide either a QdrantClient instance or a url and api_key."
            )

        if client is None and aclient is None:
            client_kwargs = client_kwargs or {}
            self._client = qdrant_client.QdrantClient(
                url=url, api_key=api_key, **client_kwargs
            )
            self._aclient = qdrant_client.AsyncQdrantClient(
                url=url, api_key=api_key, **client_kwargs
            )
        else:
            if client is not None and aclient is not None:
                logger.warning(
                    "Both client and aclient are provided. If using `:memory:` "
                    "mode, the data between clients is not synced."
                )

            self._client = client
            self._aclient = aclient

        if self._client is not None:
            self._collection_initialized = self._collection_exists(collection_name)
        else:
            #  need to do lazy init for async clients
            self._collection_initialized = False

        # TODO: setup hybrid search if enabled
        # if enable_hybrid:
        #     self._sparse_doc_fn = sparse_doc_fn or default_sparse_encoder(
        #         "naver/efficient-splade-VI-BT-large-doc"
        #     )
        #     self._sparse_query_fn = sparse_query_fn or default_sparse_encoder(
        #         "naver/efficient-splade-VI-BT-large-query"
        #     )
        #     self._hybrid_fusion_fn = hybrid_fusion_fn or cast(
        #         HybridFusionCallable, relative_score_fusion
        #     )

        super().__init__(
            client=client,
            collection_name=collection_name,
            url=url,
            api_key=api_key,
            batch_size=batch_size,
            parallel=parallel,
            max_retries=max_retries,
            client_kwargs=client_kwargs or {},
            enable_hybrid=enable_hybrid,
            sparse_doc_fn=sparse_doc_fn,
            sparse_query_fn=sparse_query_fn,
        )

    @classmethod
    def class_name(cls) -> str:
        return "MultiModalQdrantVectorStore"


    #TODO: write a more flexible hybrid fusion implementation
    def text_to_caption_query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Text-to-Text query with similarity of query text embedding and node text (image caption) embedding

        Args:
            query (VectorStoreQuery): query
        """
        
        return super().query(query, **kwargs)



    def text_to_image_query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """
        Text-to-Image query with cros-modal similarity of query text embedding and node image embedding

        Args:
            query (VectorStoreQuery): query
        """
        query_embedding = cast(List[float], query.query_embedding)
        #  NOTE: users can pass in qdrant_filters (nested/complicated filters) to override the default MetadataFilters
        qdrant_filters = kwargs.get("qdrant_filters")
        if qdrant_filters is not None:
            query_filter = qdrant_filters
        else:
            query_filter = cast(Filter, self._build_query_filter(query))

                
        response = self._client.search_batch(
            collection_name=self.collection_name,
            requests=[
                rest.SearchRequest(
                    vector=rest.NamedVector(
                        name="multi-modal",
                        vector=query_embedding,
                    ),
                    limit=query.similarity_top_k,
                    filter=query_filter,
                    with_payload=True,
                ),
            ],
        )
        return self.parse_image_to_query_result(response[0])


    def parse_image_to_query_result(self, response: List[Any]) -> VectorStoreQueryResult:
        """
        Convert vector store response to VectorStoreQueryResult.

        Args:
            response: List[Any]: List of results returned from the vector store.
        """
        nodes = []
        similarities = []
        ids = []

        for point in response:
            payload = cast(Payload, point.payload)

            # try:
            #     node = metadata_dict_to_node(payload)
            # except Exception:
            #     # NOTE: deprecated legacy logic for backward compatibility
            #     logger.debug("Failed to parse Node metadata, fallback to legacy logic.")
            #     metadata, node_info, relationships = legacy_metadata_dict_to_node(
            #         payload
            #     )

            #     node = ImageNode(
            #         id_=str(point.id),
            #         text=payload.get("text"),
            #         image=payload.get("image"),
            #         image_path=payload.get("image_path"),
            #         metadata=metadata,
            #         relationships=relationships,
            #     )

            node = ImageNode(
                    id_=str(point.id),
                    text=payload.get("text"),
                    image=payload.get("image"),
                    image_path=payload.get("image_path"),
                    metadata=payload.get("metadata"),
                    #relationships=relationships,
                )

                  


            nodes.append(node)
            similarities.append(point.score)
            ids.append(str(point.id))

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)


    #TODO
    def _build_query_filter(self, query: VectorStoreQuery) -> Optional[Any]:

        return super()._build_query_filter(query)




                