CONTEXT_PROMPT_TEMPLATE = """### Instruction to Assistant
You are an expert in computer science who is also skilled at teaching and explaining concepts. Your task is to respond to the User's current query by synthesizing information from both the text and image sources. 
Ensure that your response is coherent with the dialogue history and accurately references relevant sources. Be concise, informative, and consider both the textual and visual context. 
If you do not know the answer, reply with 'I am sorry, I dont have enough information'.

### Contextual Information
"""

DEFAULT_PROMPT_TEMPLATE = """### Instruction to Assistant
You are an expert in computer science who is also skilled at teaching and explaining concepts. Your task is to respond to the User's current query. 
Ensure that your response is coherent with the dialogue history. Be concise and informative. 
If you do not know the answer, reply with 'I am sorry, I dont have enough information'.
"""

# - **Text Sources**: 
#   {}

# - **Image Sources**: 
#   {}

IMAGE_TOKEN = "<image>./</image>"

def generate_sys_prompt(sources = None):
    if sources and (sources['text_sources'] or sources['image_sources']):
        prompt = CONTEXT_PROMPT_TEMPLATE
        if sources['text_sources']:
            prompt += "- **Text Sources**:\n"
            for i, item in enumerate(sources['text_sources']):
                text_chunk = item['text'].replace('\n', '\n  ')
                prompt += f"  - Source [{i+1}]:\n  {text_chunk}\n"
        if sources['image_sources']:
            prompt += "- **Image Sources**:\n"
            for i, item in enumerate(sources['image_sources']):
                prompt += f"  - Fig [{i+1}]: {IMAGE_TOKEN}\n  Caption: {item['caption']}\n"
    else:
        prompt = DEFAULT_PROMPT_TEMPLATE
    return prompt