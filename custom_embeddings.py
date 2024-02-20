import logging
from typing import Any, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from llama_index.legacy.bridge.pydantic import Field, PrivateAttr
from llama_index.legacy.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.legacy.embeddings.base import Embedding
from llama_index.legacy.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.legacy.schema import ImageType

logger = logging.getLogger(__name__)


class CustomizedCLIPEmbedding(MultiModalEmbedding):
    """Customized multimodal embedding models for encoding text and image for Multi-Modal purpose. (e.g. CLIP, BLIP, BLIP2)

    This class provides an interface to generate embeddings using a model
    deployed in OpenAI CLIP. At the initialization it requires a model name
    of CLIP.

    """

    embed_batch_size: int = Field(default=DEFAULT_EMBED_BATCH_SIZE, gt=0)

    _clip: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _preprocess: Any = PrivateAttr()
    _device: Any = PrivateAttr()

    @classmethod
    def class_name(cls) -> str:
        return "CustomizedCLIPEmbedding"

    def __init__(
        self,
        *,
        model_name: str,
        device: str = None,
        embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
        **kwargs: Any,
    ):
        """Initializes the ClipEmbedding class.

        Args:
            embed_batch_size (int, optional): The batch size for embedding generation. Defaults to 10,
                must be > 0 and <= 100.
            model_name (str): The model name of Clip model.

        Raises:
            ImportError: If the `clip` package is not available in the PYTHONPATH.
            ValueError: If the model cannot be fetched from Open AI. or if the embed_batch_size
                is not in the range (0, 100].
        """
        if embed_batch_size <= 0:
            raise ValueError(f"Embed batch size {embed_batch_size}  must be > 0.")

        # try:
        #     import clip
        #     import torch
        # except ImportError:
        #     raise ImportError(
        #         "ClipEmbedding requires `pip install git+https://github.com/openai/CLIP.git` and torch."
        #     )

        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
        except ImportError:
            raise ImportError(
                "CustomizedCLIPEmbedding requires huggingface transformers and torch."
            )

        super().__init__(
            embed_batch_size=embed_batch_size, model_name=model_name, **kwargs
        )

        # try:
        #     self._device = "cuda" if torch.cuda.is_available() else "cpu"
        #     if self.model_name not in AVAILABLE_CLIP_MODELS:
        #         raise ValueError(
        #             f"Model name {self.model_name} is not available in CLIP."
        #         )
        #     self._model, self._preprocess = clip.load(
        #         self.model_name, device=self._device
        #     )

        # except Exception as e:
        #     logger.error(f"Error while loading clip model.")
        #     raise ValueError("Unable to fetch the requested embeddings model") from e

        try:
            if device == None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = device
            self._model = CLIPModel.from_pretrained(self.model_name).to(self._device)
            self._preprocess = CLIPProcessor.from_pretrained(self.model_name)

        except Exception as e:
            logger.error(f"Error while loading clip model.")
            raise ValueError("Unable to fetch the requested embeddings model") from e



    # TEXT EMBEDDINGS

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        results = []
        try:
            import torch
        except ImportError:
            raise ImportError(
                "CustomizedCLIPEmbedding requires `pip install torch`."
            )
        with torch.no_grad():
            for text in texts:
                # try:
                #     import clip
                # except ImportError:
                #     raise ImportError(
                #         "ClipEmbedding requires `pip install git+https://github.com/openai/CLIP.git` and torch."
                #     )
                # text_embedding = self._model.encode_text(
                #     clip.tokenize(text).to(self._device)
                # )

                #TODO
                text_embedding = self._model.get_text_features(**self._preprocess.tokenizer(text, return_tensors="pt").to(self._device))

                results.append(text_embedding.tolist()[0])

        return results

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._get_text_embedding(query)

    # IMAGE EMBEDDINGS

    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        return self._get_image_embedding(img_file_path)

    def _get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        try:
            import torch
            from PIL import Image
        except ImportError:
            raise ImportError(
                "CustomizedCLIPEmbedding requires `pip install torch` and `pip install pillow`."
            )
        # with torch.no_grad():
        #     image = (
        #         self._preprocess(Image.open(img_file_path))
        #         .unsqueeze(0)
        #         .to(self._device)
        #     )
        #     return self._model.encode_image(image).tolist()[0]
        with torch.no_grad():
            img_inputs = self._preprocess.image_processor.preprocess(Image.open(img_file_path), return_tensors="pt").to(self._device)
            return self._model.get_image_features(**img_inputs).tolist()[0]





def custom_sparse_doc_vectors(
    doc_tokenizer,
    doc_model,
    max_length: int,
    texts: List[str],
) -> Tuple[List[List[int]], List[List[float]]]:

    tokens = doc_tokenizer(
        texts, max_length=max_length, truncation=True, padding=True, return_tensors="pt"
    )

    # if torch.cuda.is_available():
    #     tokens = tokens.to("cuda")

    output = doc_model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    tvecs, _ = torch.max(weighted_log, dim=1)

    indices = []
    vecs = []
    for batch in tvecs:
        indices.append(batch.nonzero(as_tuple=True)[0].tolist())
        vecs.append(batch[indices[-1]].tolist())

    return indices, vecs


def custom_sparse_query_vectors(
    query_tokenizer,
    query_model,
    max_length: int,
    texts: List[str],
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Computes vectors from logits and attention mask using ReLU, log, and max operations.
    """
    # TODO: compute sparse vectors in batches if max length is exceeded
    tokens = query_tokenizer(
        texts, max_length=max_length, truncation=True, padding=True, return_tensors="pt"
    )
    
    # if torch.cuda.is_available():
    #     tokens = tokens.to("cuda")

    output = query_model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    tvecs, _ = torch.max(weighted_log, dim=1)

    # extract the vectors that are non-zero and their indices
    indices = []
    vecs = []
    for batch in tvecs:
        indices.append(batch.nonzero(as_tuple=True)[0].tolist())
        vecs.append(batch[indices[-1]].tolist())

    return indices, vecs


if __name__ == "__main__":

    import requests
    from PIL import Image

    CLIP_PATH = "./embedding_models/clip-vit-base-patch32"
    clip_embedding = CustomizedCLIPEmbedding(model_name=CLIP_PATH, device="cpu")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    texts = ["This is a test sentence for custom CLIP embedding model.", "This is another test sentence."]
    txt_embedding = clip_embedding._get_text_embeddings(texts)
    print(f"\n\nText Embedding: {txt_embedding}")

    img_embedding = clip_embedding._get_image_embedding(requests.get(url, stream=True).raw)
    print(f"\n\nImage Embedding: {img_embedding}")