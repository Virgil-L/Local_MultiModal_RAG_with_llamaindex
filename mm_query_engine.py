from typing import Any, Dict, List, Optional, Sequence, Tuple

from llama_index.core.base.response.schema import RESPONSE_TYPE, Response
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.indices.multi_modal import MultiModalVectorIndexRetriever
from llama_index.core.indices.query.base import BaseQueryEngine
from llama_index.core.indices.query.schema import QueryBundle, QueryType
from llama_index.core.multi_modal_llms.base import MultiModalLLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_TEXT_QA_PROMPT
from llama_index.core.query_engine.citation_query_engine import CITATION_QA_TEMPLATE
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.schema import ImageNode, NodeWithScore
from mm_retriever import MultiModalQdrantRetriever


# rewrite CITATION_QA_TEMPLATE
TEXT_QA_TEMPLATE = PromptTemplate(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "Every answer should include at least one source citation. "
    "Only cite a source when you are explicitly referencing it. "
    "If none of the sources are helpful, you should indicate that. "
    "Below are several numbered sources of information:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

IMAGE_QA_TEMPLATE = PromptTemplate(
    "<image>\n"
    "Caption: {context_str}"
    "\n------\n"
    "You are a smart agent who can answer questions based on external information. "
    "Above is an annotated image you retrieved. Please provide an answer to the query based solely on the image and caption. "
    "If the image is not helpful, you should indicate that. \n"
    "Query: {query_str}\n"
    "Note: Don't include expressions like \"This image appears to be XXX\" in your answer.\n"
    "Answer: "
)

ANSWER_INTEGRATION_TEMPLATE = PromptTemplate(
    "With the following sources related to your question from my knowledge base: \n"
    "\n"+"-"*50+"\n"
    "Paragraphs:\n\n"
    "{context_str}\n"
    "\nImages:\n"
    "{image_context_str}\n"
    "\n"+"-"*50+"\n"
    "Here is my answer:\n"
    "\n{text_context_response}\n{image_context_response}"
)

# def _get_image_and_text_nodes(
#     nodes: List[NodeWithScore],
# ) -> Tuple[List[NodeWithScore], List[NodeWithScore]]:
#     image_nodes = []
#     text_nodes = []
#     for res_node in nodes:
#         if isinstance(res_node.node, ImageNode):
#             image_nodes.append(res_node)
#         else:
#             text_nodes.append(res_node)
#     return image_nodes, text_nodes


class CustomMultiModalQueryEngine(BaseQueryEngine):
    """Simple Multi Modal Retriever query engine.

    Assumes that retrieved text context fits within context window of LLM, along with images.

    Args:
        retriever (MultiModalVectorIndexRetriever): A retriever object.
        multi_modal_llm (Optional[MultiModalLLM]): MultiModalLLM Models.
        text_qa_template (Optional[BasePromptTemplate]): Text QA Prompt Template.
        image_qa_template (Optional[BasePromptTemplate]): Image QA Prompt Template.
        node_postprocessors (Optional[List[BaseNodePostprocessor]]): Node Postprocessors.
        callback_manager (Optional[CallbackManager]): A callback manager.
    """

    def __init__(
        self,
        retriever: MultiModalQdrantRetriever,
        multi_modal_llm: MultiModalLLM,
        text_qa_template: Optional[BasePromptTemplate] = None,
        image_qa_template: Optional[BasePromptTemplate] = None, 
        answer_integration_template: Optional[BasePromptTemplate] = None,
        node_postprocessors: Optional[List[BaseNodePostprocessor]] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs: Any,
    ) -> None:
        self._retriever = retriever
        self._multi_modal_llm = multi_modal_llm

        self._text_qa_template = text_qa_template or CITATION_QA_TEMPLATE
        self._image_qa_template = image_qa_template or IMAGE_QA_TEMPLATE

        self._answer_integration_template = answer_integration_template or ANSWER_INTEGRATION_TEMPLATE

        self._node_postprocessors = node_postprocessors or []
        callback_manager = callback_manager or CallbackManager([])
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = callback_manager

        super().__init__(callback_manager)

    def _get_prompts(self) -> Dict[str, Any]:
        """Get prompts."""
        return {"text_qa_template": self._text_qa_template}

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {}

    def _apply_node_postprocessors(
        self, nodes: List[NodeWithScore], query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        for node_postprocessor in self._node_postprocessors:
            nodes = node_postprocessor.postprocess_nodes(
                nodes, query_bundle=query_bundle
            )
        return nodes

    def retrieve(self, 
                query_bundle: QueryBundle, 
                text_query_mode: str = "hybrid",
                image_query_mode: str = "default",
                metadata_filters = None) -> Dict[str, List[NodeWithScore]]:

        text_retrieval_result = self._retriever.retrieve_text_nodes(query_bundle, text_query_mode, metadata_filters)
        image_retrieval_result = self._retriever.retrieve_image_nodes(query_bundle, image_query_mode, metadata_filters)

        reranked_text_nodes = self._retriever.rerank_text_nodes(query_bundle, text_retrieval_result)
        reranked_image_nodes = self._retriever.rerank_image_nodes(query_bundle, image_retrieval_result)

        retrieval_results = {
            "text_nodes": self._apply_node_postprocessors(reranked_text_nodes, query_bundle=query_bundle),
            "image_nodes": self._apply_node_postprocessors(reranked_image_nodes, query_bundle=query_bundle),
        }

        return retrieval_results

    # async def aretrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
    #     nodes = await self._retriever.aretrieve(query_bundle)
    #     return self._apply_node_postprocessors(nodes, query_bundle=query_bundle)

    def synthesize(
        self,
        query_bundle: QueryBundle,
        #nodes: List[NodeWithScore],
        retrieval_results: Dict[str, List[NodeWithScore]],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:


        #image_nodes, text_nodes = _get_image_and_text_nodes(nodes)
        image_nodes, text_nodes = retrieval_results["image_nodes"], retrieval_results["text_nodes"]
        
        #TODO: format prompt with (text context), (image + caption of image)
        context_str = "\n\n".join([f"Source {text_nodes.index(r)+1}:\n" + r.get_content() for r in text_nodes])
        fmt_prompt = self._text_qa_template.format(
            context_str=context_str, 
            query_str=query_bundle.query_str,
        )

        image_context_str = "\n\n".join([r.get_content() for r in image_nodes])
        image_query_fmt_prompt = self._image_qa_template.format(context_str=image_context_str, query_str=query_bundle.query_str)

        text_context_messages = [
            {
                "role": "user",
                "content":[
                    {"type":"text", "text":fmt_prompt}
                ]
            }
        ]


        ## Generate response when the mllm(llava) is under llamacpp framework
        ##TODO: handle multiple image input
        image_url = f"data:image/png;base64,{image_nodes[0].node.image.decode('utf-8')}"
        image_context_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": image_query_fmt_prompt}
                ]
            }
        ]
        text_context_response = self._multi_modal_llm.chat(
            messages=text_context_messages,
        )

        image_context_response = self._multi_modal_llm.chat(
            messages=image_context_messages,
        )


        ## Generate response when the mllm(llava) is under ollama framework
        # text_context_response = self._multi_modal_llm.complete(
        #     prompt=fmt_prompt,
        #     images=[],
        # )

        # image_context_response = self._multi_modal_llm.complete(
        #     prompt=image_query_fmt_prompt,
        #     images=[image_node.node.image for image_node in image_nodes],
        # )


        #TODO: transform encoded base64 image to image object in GUI
        synthesized_response = self._answer_integration_template.format(
            context_str=context_str,
            image_context_str= "\n\n".join(["<image>"+ str(r.node.image) + '</image>\n' + r.node.get_content() for r in image_nodes]),
            text_context_response=text_context_response.text.replace("\n"," ").strip(),
            image_context_response=i_q_response.text.replace("\n"," ").strip(),
            )

        return Response(
            response=str(synthesized_response),
            source_nodes=text_nodes+image_nodes,
            metadata={
                "query_str": query_bundle.query_str,
                "model_config": self._multi_modal_llm.metadata,
            },
        )



    # async def asynthesize(
    #     self,
    #     query_bundle: QueryBundle,
    #     nodes: List[NodeWithScore],
    #     additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    # ) -> RESPONSE_TYPE:
    #     image_nodes, text_nodes = _get_image_and_text_nodes(nodes)
    #     context_str = "\n\n".join([r.get_content() for r in text_nodes])
    #     fmt_prompt = self._text_qa_template.format(
    #         context_str=context_str, query_str=query_bundle.query_str
    #     )
    #     llm_response = await self._multi_modal_llm.acomplete(
    #         prompt=fmt_prompt,
    #         image_documents=image_nodes,
    #     )
    #     return Response(
    #         response=str(llm_response),
    #         source_nodes=nodes,
    #         metadata={"text_nodes": text_nodes, "image_nodes": image_nodes},
    #     )
    
    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: List[NodeWithScore],
        additional_source_nodes: Optional[Sequence[NodeWithScore]] = None,
    ) -> RESPONSE_TYPE:
        raise NotImplementedError("Async synthesize not implemented yet")


    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            with self.callback_manager.event(
                CBEventType.RETRIEVE,
                payload={EventPayload.QUERY_STR: query_bundle.query_str},
            ) as retrieve_event:
                retrieval_results = self.retrieve(query_bundle)

                retrieve_event.on_end(
                    payload={EventPayload.NODES: retrieval_results},
                )

            response = self.synthesize(
                query_bundle,
                retrieval_results=retrieval_results,
            )

            query_event.on_end(payload={EventPayload.RESPONSE: response})

        return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        raise NotImplementedError("Async query not implemented yet")

    # async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
    #     """Answer a query."""
    #     with self.callback_manager.event(
    #         CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
    #     ) as query_event:
    #         with self.callback_manager.event(
    #             CBEventType.RETRIEVE,
    #             payload={EventPayload.QUERY_STR: query_bundle.query_str},
    #         ) as retrieve_event:
    #             nodes = await self.aretrieve(query_bundle)

    #             retrieve_event.on_end(
    #                 payload={EventPayload.NODES: nodes},
    #             )

    #         response = await self.asynthesize(
    #             query_bundle,
    #             nodes=nodes,
    #         )

    #         query_event.on_end(payload={EventPayload.RESPONSE: response})

    #     return response


    @property
    def retriever(self) -> MultiModalVectorIndexRetriever:
        """Get the retriever object."""
        return self._retriever
