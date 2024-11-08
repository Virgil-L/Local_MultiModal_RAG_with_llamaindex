import os
import streamlit as st
from PIL import Image
import argparse
from streamlit_feedback import streamlit_feedback
import time
from io import BytesIO
import base64
import requests
from tqdm import tqdm
import json
import ollama
from src.prompt import generate_sys_prompt

MLLM_NAME = 'minicpm-v:8b-2.6-q4_K_M'
DEFAULT_PROLOGUE = "Hello! I'm a multimodal retriever. I can help you with a variety of tasks. What would you like to know?"

st.set_page_config(
    page_title="Chat with Multimodal Retriever",
    page_icon="🔍",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/Virgil-L",
        "Report a bug": "https://github.com/Virgil-L",
        "About": "Built by @Virgil-L with Streamlit & LlamaIndex",
    }
)

# Initialize session state
if 'llm_prompt_tokens' not in st.session_state:
    st.session_state['llm_prompt_tokens'] = 0

if 'llm_completion_tokens' not in st.session_state:
    st.session_state['llm_completion_tokens'] = 0

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'btn_llama_index' not in st.session_state:
    st.session_state['btn_llama_index'] = False

if 'btn_retriever' not in st.session_state:
    st.session_state['btn_retriever'] = False

if 'btn_diff' not in st.session_state:
    st.session_state['btn_diff'] = False

if 'btn_rag' not in st.session_state:
    st.session_state['btn_rag'] = False
# if 'openai_api_key' in st.session_state:
#     openai.api_key = st.session_state['openai_api_key']

def mm_retrieve(query, text_topk = 3, image_topk = 1, port=5000):
    req_data = {"query": query, "text_topk": text_topk, "image_topk": image_topk}

    response = requests.post(f"http://localhost:{port}/api", headers={"Content-Type": "application/json"}, data=json.dumps(req_data))
    rep_data = response.json()
    text_sources, image_sources = [], []
    if 'text_result' in rep_data:
        text_sources = [{'text': item['node']['text'], 
                         'elementType': item['node']['metadata']['metadata']['elementType'],
                         'source_file': item['node']['metadata']['metadata']['source_file_path']} for item in rep_data['text_result']]
    if 'image_result' in rep_data:
        image_sources = [{'image': base64.b64decode(item['node']['image']), 
                          'caption': item['node']['text'],
                          'elementType': item['node']['metadata']['elementType'],
                          'source_file': item['node']['metadata']['source_file_path']} for item in rep_data['image_result']]
    return {
        'text_sources': text_sources,
        'image_sources': image_sources
    }


def display_sources(sources):
    with st.expander("See sources"):
        if sources['text_sources']:
            st.markdown("#### Text sources:")
            for i, item in enumerate(sources['text_sources']):
                # 对每一篇文章，创建一个container，展示文章的文本内容
                txt_sources_container = st.container()
                with txt_sources_container:
                    st.markdown(f"**Ref [{i+1}]**:\n\nfrom: {item['source_file']}\n\n > {item['text']}\n\n")
        if sources['image_sources']:
            st.markdown("#### Image sources:")
            for i, item in enumerate(sources['image_sources']):
                img_sources_container = st.container()
                with img_sources_container:
                    st.markdown(f"**Ref [{i+1}]**\n\nfrom: {(item['source_file'])} \n\n")
                    st.image(item['image'], caption=item['caption'], use_column_width=True)


def display_chat_history(messages, dialogue_container):
    """Display previous chat messages."""
    with dialogue_container:
        for message in messages:
            with st.chat_message(message["role"]):
                if st.session_state.with_sources:
                    if "sources" in message:
                        #TODO: 展示图片，文本块等参考信息
                        display_sources(message["sources"])
                st.write(message["content"])    
            
            


def clear_chat_history():
    """"Clear chat history and reset questions' buttons."""

    # st.session_state.messages = [
    #         {"role": "assistant", "content": DEFAULT_PROLOGUE}
    #     ]
    st.session_state.messages = []
    st.session_state["btn_llama_index"] = False
    st.session_state["btn_retriever"] = False
    st.session_state["btn_diff"] = False
    st.session_state["btn_rag"] = False



def upload_image():
    uploaded_file = st.file_uploader("Choose an image from your computer", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        return image



def get_user_input():
    query = st.chat_input(placeholder="Enter your prompt here", max_chars=2048)
    if query:
        st.session_state['llm_prompt_tokens'] += 1
        st.session_state['messages'].append({"role": "user", "content": query})
    return query



def request_assistant_response(messages, sources=None):
    ##TODO: prompt模板、多模态上下文输入、多轮对话
    system = generate_sys_prompt(sources)

    if messages[0]['role'] == 'assistant':
        messages = messages[1:]
    if sources and sources['image_sources']:
        messages.insert(0, {'role': 'system', 'content': system, 'images': [item['image'] for item in sources['image_sources']]})
    else:
        messages.insert(0, {'role': 'system', 'content': system})
    resp_stream = ollama.chat(
            model=MLLM_NAME,
            messages=messages,
            stream=True,
            keep_alive=600,
            options={
                'num_predict': -1,
                'temperature': st.session_state.get('temperature', 0.5),
                'top_p': st.session_state.get('top_p', 0.9),
                'stop': ['<EOT>', '<|im_end|>'],
                'frequency_penalty':st.session_state.get('frequency_penalty', 2.0),
                'num_ctx':8192,
            },
        )
    for chunk in resp_stream:
        yield chunk['message']['content']


def generate_assistant_response(messages, sources=None):

    with st.spinner("I am on it..."):
        resp_stream = request_assistant_response(messages, sources)
        with st.chat_message("assistant"):
            full_response = st.write_stream(resp_stream)
    message = {'role': 'assistant', 'content': full_response}
    if st.session_state.with_sources:
        message['sources'] = sources
        display_sources(sources)
    st.session_state.messages.append(message)



def format_sources(response):
    ##TODO
    # """Format filename, authors and scores of the response source nodes."""
    # base = "https://github.com/jerryjliu/llama_index/tree/main/"
    # return "\n".join([f"- {base}{source['filename']} (author: '{source['author']}'; score: {source['score']})\n" for source in get_metadata(response)])
    raise NotImplementedError

def get_metadata(response):
    """Parse response source nodes and return a list of dictionaries with filenames, authors and scores.""" 
    ##TODO
    # sources = []
    # for item in response.source_nodes:
    #     if hasattr(item, "metadata"):
    #         filename = item.metadata.get('filename').replace('\\', '/')
    #         author = item.metadata.get('author')
    #         score = float("{:.3f}".format(item.score))
    #         sources.append({'filename': filename, 'author': author, 'score': score})
    
    # return sources
    raise NotImplementedError



def side_bar():
    """Configure the sidebar and user's preferences."""
    st.sidebar.title("Configurations")

    with st.sidebar.expander("🔑 OPENAI-API-KEY", expanded=True):
        st.text_input(label='OPENAI-API-KEY', 
                      type='password', 
                      key='openai_api_key', 
                      label_visibility='hidden').strip()
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

    with st.sidebar.expander("💲 GPT3.5 INFERENCE COST", expanded=True):
        i_tokens = st.session_state['llm_prompt_tokens']
        o_tokens = st.session_state['llm_completion_tokens']
        st.markdown(f'LLM Prompt: {i_tokens} tokens')
        st.markdown(f'LLM Completion: {o_tokens} tokens')

        i_cost = (i_tokens / 1000) * 0.0015
        o_cost = (o_tokens / 1000) * 0.002
        st.markdown('**Cost Estimation: ${0}**'.format(round(i_cost + o_cost, 5)))
        "[OpenAI Pricing](https://openai.com/pricing)"

    with st.sidebar.expander("🔧 SETTINGS", expanded=True):
        st.toggle('Cache Results', value=True, key="with_cache")
        st.toggle('Display Sources', value=True, key="with_sources")
        st.toggle('Streaming', value=False, key="with_streaming")
        st.toggle('Debug Info', value=False, key="debug_mode")
    
    st.sidebar.button('Clear Messages', type="primary", on_click=clear_chat_history) 

    if st.session_state.debug_mode:
        with st.sidebar.expander(" 🕸️ Current Session State", expanded=True):
            st.write(st.session_state)
    

    ## Show external links

    #st.sidebar.divider()
    # with st.sidebar:
    #     col_ll, col_gh = st.columns([1, 1])
    #     with col_ll:
    #         "[![LlamaIndex Docs](https://img.shields.io/badge/LlamaIndex%20Docs-gray)](https://gpt-index.readthedocs.io/en/latest/index.html)"
    #     with col_gh:
    #         "[![Github](https://img.shields.io/badge/Github%20Repo-gray?logo=Github)](https://github.com/dcarpintero/llamaindexchat)"

    

    

def layout():
    st.header("Chat with 🦙 LlamaIndex Docs 🗂️")
    

    # Sample Questions for User input
    user_input_button = None

    ##TODO: Modify the sample questions
    btn_llama_index = st.session_state.get("btn_llama_index", False)
    btn_retriever = st.session_state.get("btn_retriever", False)
    btn_diff = st.session_state.get("btn_diff", False)
    btn_rag = st.session_state.get("btn_rag", False)

    col1, col2, col3, col4 = st.columns([1,1,1,1])

    with col1:
        if st.button("explain the basic usage pattern of LlamaIndex", type="primary", disabled=btn_llama_index):
            user_input_button = "explain the basic usage pattern in LlamaIndex"
            st.session_state.btn_llama_index = True
    with col2:
        if st.button("how can I ingest data from the GoogleDocsReader?", type="primary", disabled=btn_retriever):
            user_input_button = "how can I ingest data from the GoogleDocsReader?"
            st.session_state.btn_retriever = True
    with col3:
        if st.button("what's the difference between document & node?", type="primary", disabled=btn_diff):
            user_input_button = "what's the difference between document and node?"
            st.session_state.btn_diff = True
    with col4:
        if st.button("how can I make a RAG application performant?", type="primary", disabled=btn_rag):
            user_input_button = "how can I make a RAG application performant?"
            st.session_state.btn_rag = True


    # System Message
    if "messages" not in st.session_state or not st.session_state.messages:    
        st.session_state.messages = [
            {"role": "assistant", "content": "Try one of the sample questions or ask your own!"}
        ]

    dialogue_container = st.container()

    # User input
    user_input = st.chat_input("Enter your question here")
    if user_input or user_input_button:
        st.session_state.messages.append({"role": "user", "content": user_input or user_input_button})

    # Display previous chat
    display_chat_history(st.session_state.messages, dialogue_container)

    # Generate response
    if st.session_state.messages[-1]["role"] != "assistant":
        try:
            sources = None
            if st.session_state.with_sources:
                sources = mm_retrieve(user_input or user_input_button)
            generate_assistant_response(st.session_state.messages, sources)

        except Exception as ex:
            st.error(str(ex))


@st.cache_resource
def initialization():
    with st.spinner('Loading Environment...'):
        try:
            ##TODO: check retriever and ollama service, if not running, start them
            time.sleep(3)
            pass

        except Exception as ex:
            st.error(str(ex))
            st.stop()
            
def main():
    # Initializations
    initialization()
    side_bar()
    layout()
    

    
if __name__ == "__main__":
    main()