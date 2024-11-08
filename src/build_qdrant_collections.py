import os
import json

from qdrant_client import QdrantClient
from qdrant_client.http import models as qd_models

from llama_index.legacy.schema import ImageNode, TextNode, NodeRelationship, RelatedNodeInfo
from io import BytesIO
import base64
from PIL import Image

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from llama_index.legacy.embeddings import HuggingFaceEmbedding
from custom_embeddings import CustomizedCLIPEmbedding

from llama_index.legacy.node_parser import SentenceSplitter

from typing import (
    Any,
    Dict,
    List,
    Tuple,
)

import argparse


PDF_FOLDER = "data/paper_PDF"
BGE_PATH = "./embedding_models/bge-small-en-v1.5"
CLIP_PATH = "./embedding_models/clip-vit-base-patch32"
SPLADE_DOC_PATH = "./embedding_models/efficient-splade-VI-BT-large-doc"
SPLADE_QUERY_PATH = "./embedding_models/efficient-splade-VI-BT-large-doc"
QDRANT_PATH = "./qdrant_db"

parser = argparse.ArgumentParser(description='Parse PDFs to extract figures and texts')
parser.add_argument('--pdf_folder', type=str, default=PDF_FOLDER, help='Path to the folder containing PDFs')
parser.add_argument('--text_embedding_model', type=str, default=BGE_PATH, help='Path to the text embedding model')
parser.add_argument('--sparse_text_embedding_model', type=str, default=SPLADE_DOC_PATH, help='Path to the sparse text embedding model')
parser.add_argument('--image_embedding_model', type=str, default=CLIP_PATH, help='Path to the image embedding model')
parser.add_argument('--chunk_size', type=int, default=384, help='Size of the chunks to split the text into')
parser.add_argument('--chunk_overlap', type=int, default=32, help='Overlap between the chunks')
parser.add_argument('--storage_path', type=str, default=QDRANT_PATH, help='Path of the qdrant storage')






def extract_text_nodes(text_data: Dict, text_parser, config_file) -> List[TextNode]:
    title_node = TextNode(
        text = ':\n'.join(('title', text_data['title'])),
        metadata = {
            "source_file_path": os.path.join(os.getcwd(), PDF_FOLDER, config_file.replace(".json", ".pdf")),
            "elementType": "title",
        }
    )

    author_node = TextNode(
        text = ':\n'.join(('authors', text_data['authors'])),
        metadata = {
            "source_file_path": os.path.join(os.getcwd(), PDF_FOLDER, config_file.replace(".json", ".pdf")),
            "elementType": "author",
        }
    )

    abstract_text = ':\n'.join(('abstract', text_data['abstract']))
    splitted_abstract = text_parser.split_text(abstract_text)
    abstract_nodes = [TextNode(
            text = text, metadata = {"source_file_path": os.path.join(os.getcwd(), PDF_FOLDER, config_file.replace(".json", ".pdf")),"elementType": "abstract",}
        ) for text in splitted_abstract]


    section_text_list = [section['heading']+'\n'+section['text'] for section in text_data['sections']]


    for i in range(len(section_text_list)-1, -1, -1):
        if len(section_text_list[i]) < text_parser.chunk_size:
            if i > 0:
                section_text_list[i-1] += "\n" + section_text_list[i]
                section_text_list.pop(i)
            else:
                section_text_list[i+1] += "\n" + section_text_list[i]
                section_text_list.pop(i)

    
    section_nodes = []
    for section_text in section_text_list:
        splitted_section = text_parser.split_text(section_text)
        section_nodes.extend([TextNode(
            text = text, metadata = {"source_file_path": os.path.join(os.getcwd(), PDF_FOLDER, config_file.replace(".json", ".pdf")),"elementType": "section",}
        ) for text in splitted_section])


    non_title_nodes = [author_node] + abstract_nodes + section_nodes
    for node in non_title_nodes:
        build_parent_child_relationships(title_node, node)

    return [title_node] + non_title_nodes



def build_parent_child_relationships(parent_node, child_node):
    child_node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(node_id=parent_node.id_, metadata={'elementType':parent_node.metadata['elementType']})

    if NodeRelationship.CHILD not in parent_node.relationships.keys():
        parent_node.relationships[NodeRelationship.CHILD] = [
            RelatedNodeInfo(
                node_id=child_node.id_, 
                metadata={'elementType':child_node.metadata['elementType']}
            )
        ]
    else:
        parent_node.relationships[NodeRelationship.CHILD].append(
            RelatedNodeInfo(
                node_id=child_node.id_, 
                metadata={'elementType':child_node.metadata['elementType']}
            )
        )
    
    return



def compute_sparse_text_vector(text, tokenizer, model, max_length=512):
    """
    Computes a vector from logits and attention mask using ReLU, log, and max operations.
    """

    tokens = tokenizer(
        text, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
    output = model(**tokens)
    logits, attention_mask = output.logits, tokens.attention_mask
    relu_log = torch.log(1 + torch.relu(logits))
    weighted_log = relu_log * attention_mask.unsqueeze(-1)
    max_val, _ = torch.max(weighted_log, dim=1)
    vec = max_val.squeeze()

    indices = vec.nonzero(as_tuple=True)[0].tolist()
    values = vec[indices].tolist()
    
    return qd_models.SparseVector(indices=indices, values=values)



def build_text_nodes(chunk_size, chunk_overlap, pdf_folder, bge_embedding):
    TEXT_FOLDER = os.path.join(PDF_FOLDER, "parsed_texts")
    text_config_files = os.listdir(TEXT_FOLDER)

    text_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # separator=" ",
    )

    text_nodes = []
    for config_file in text_config_files:
        if config_file.endswith(".json"):
            with open(os.path.join(TEXT_FOLDER, config_file), "r") as cf:
                text_data = json.load(cf)
            text_nodes.extend(extract_text_nodes(text_data, text_parser, config_file))
        

    for text_node in text_nodes:
        text_embedding = bge_embedding.get_text_embedding(text_node.get_text())
        text_node.embedding = text_embedding

    return text_nodes


def create_text_collection(text_nodes, client, collection_name, sparse_tokenizer, sparse_embedding):
    text_embedding_size = len(text_nodes[0].embedding)

    try:
        client.get_collection(collection_name)
    except:
        client.create_collection(
        collection_name=collection_name,
            vectors_config={
                "text-dense": qd_models.VectorParams(size=text_embedding_size, distance=qd_models.Distance.COSINE, on_disk=True),
            },
            sparse_vectors_config={
                "text-sparse": qd_models.SparseVectorParams(
                    index=qd_models.SparseIndexParams(on_disk=True)
                )
            },
            optimizers_config=qd_models.OptimizersConfigDiff(memmap_threshold=20000),
        )
    
    client.upsert(
        collection_name=collection_name,
        points=[
            qd_models.PointStruct(
                id = text_node.id_,
                vector = {
                    "text-dense": text_node.embedding,
                    "text-sparse": compute_sparse_text_vector(text_node.get_text(), sparse_tokenizer, sparse_embedding)
                },
                payload = {
                    "text": text_node.text,
                    "metadata": text_node.metadata,
                }
            ) for text_node in text_nodes
        ]
    )

    return



def build_image_nodes(pdf_folder, bge_embedding, clip_embedding):

    IMAGE_FOLDER = os.path.join(pdf_folder, "parsed_figures", "data")
    img_config_files = os.listdir(IMAGE_FOLDER)

    img_nodes = []

    for config_file in img_config_files:
        if config_file.endswith(".json"):
            with open(os.path.join(IMAGE_FOLDER, config_file), "r") as cf:
                img_data = json.load(cf)

            for img_config in img_data:
                with open(img_config['renderURL'], "rb") as img_file:
                    img_base64_bytes = base64.b64encode(img_file.read())

                img_metadata = {k: img_config[k] for k in img_config.keys() & {'name', 'page', 'figType', 'imageText', 'regionBoundary',  'captionBoundary'} }
                img_metadata['elementType'] = img_metadata.pop('figType')
                img_metadata['source_file_path'] = os.path.join(os.getcwd(), PDF_FOLDER, config_file.replace(".json", ".pdf"))

                img_node = ImageNode(
                    image = img_base64_bytes,
                    metadata = img_metadata,
                    image_path=img_config["renderURL"],
                    text=img_config['caption'],
                )    
                img_nodes.append(img_node)

    for img_node in img_nodes:
        img_embedding = clip_embedding.get_image_embedding(BytesIO(base64.b64decode(img_node.image)))
        text_embedding = bge_embedding.get_text_embedding(img_node.text)
        img_node.embedding = img_embedding
        img_node.text_embedding = text_embedding
       

    return img_nodes



def create_image_collection(image_nodes, client, collection_name, sparse_tokenizer, sparse_embedding):
    image_embedding_size = len(image_nodes[0].embedding)
    text_embedding_size = len(image_nodes[0].text_embedding)

    try:
        client.get_collection(collection_name)
    except:
        client.create_collection(
        collection_name=collection_name,
            vectors_config={
                "multi-modal": qd_models.VectorParams(size=image_embedding_size, distance=qd_models.Distance.COSINE, on_disk=True),
                "text-dense": qd_models.VectorParams(size=text_embedding_size, distance=qd_models.Distance.COSINE, on_disk=True),
            },
            sparse_vectors_config={
                "text-sparse": qd_models.SparseVectorParams(
                    index=qd_models.SparseIndexParams(on_disk=True)
                )
            },
            optimizers_config=qd_models.OptimizersConfigDiff(memmap_threshold=20000),
        )

    client.upsert(
        collection_name=collection_name,
        points=[
            qd_models.PointStruct(
                id = image_node.id_,
                vector = {
                    "multi-modal": image_node.embedding,
                    "text-dense": image_node.text_embedding,
                    "text-sparse":compute_sparse_text_vector(image_node.text, sparse_tokenizer, sparse_embedding),
                },

                payload = {
                    "image_path": image_node.image_path,
                    "metadata": image_node.metadata,
                    "image": image_node.image,
                    "text": image_node.text,
                }
            ) for image_node in image_nodes
        ]
    )

    return    



if __name__ == "__main__":
    args = parser.parse_args()

    print("Loading Embedding Models...\n")
    bge_embedding = HuggingFaceEmbedding(model_name=args.text_embedding_model, device="cpu", pooling="mean")
    clip_embedding = CustomizedCLIPEmbedding(model_name=args.image_embedding_model, device="cpu")

    splade_doc_tokenizer = AutoTokenizer.from_pretrained(args.sparse_text_embedding_model)
    splade_doc_embedding = AutoModelForMaskedLM.from_pretrained(args.sparse_text_embedding_model)

    print("Building Text and Image Nodes...\n")

    text_nodes = build_text_nodes(
        chunk_size=args.chunk_size, 
        chunk_overlap=args.chunk_overlap, 
        pdf_folder=args.pdf_folder, 
        bge_embedding=bge_embedding, 
        )

    image_nodes = build_image_nodes(  
        pdf_folder = args.pdf_folder, 
        bge_embedding = bge_embedding, 
        clip_embedding = clip_embedding,
        )

    print("Creating Qdrant Collections...\n")
    client = QdrantClient(path=args.storage_path)
    create_text_collection(
        text_nodes=text_nodes, 
        client=client, 
        collection_name="text_collection", 
        sparse_tokenizer=splade_doc_tokenizer, 
        sparse_embedding=splade_doc_embedding)

    create_image_collection(
        image_nodes=image_nodes, 
        client=client, 
        collection_name="image_collection",
        sparse_tokenizer=splade_doc_tokenizer, 
        sparse_embedding=splade_doc_embedding
        )






    
