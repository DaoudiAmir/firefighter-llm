import os
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set page configuration
st.set_page_config(
    page_title="Firefighter Assistant AI",
    page_icon="🚒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model paths
MODEL_PATH = "models/firefighter-assistant-v1/final_model"
BEST_MODEL_PATH = "models/firefighter-assistant-v1/best_model"

# Predefined prompts
PREDEFINED_PROMPTS = {
    "Fire Safety Protocol": {
        "instruction": "Explain the standard protocol for responding to a residential fire.",
        "input": "A two-story house with potential occupants inside."
    },
    "Chemical Hazard": {
        "instruction": "What are the safety procedures for handling a chemical spill?",
        "input": "Industrial accident involving unknown chemicals with strong odor."
    },
    "First Aid": {
        "instruction": "Provide first aid instructions for burn victims.",
        "input": "Second-degree burns on arms and hands."
    },
    "Evacuation Plan": {
        "instruction": "Outline an effective evacuation plan for a multi-story building.",
        "input": "Office building with 200 occupants during working hours."
    },
    "Equipment Usage": {
        "instruction": "Explain how to properly use a fire extinguisher.",
        "input": "Class B fire in a kitchen environment."
    },
    "Emergency Communication": {
        "instruction": "What information should be communicated when reporting a fire?",
        "input": "Witnessing a fire at a neighbor's house."
    },
    "Wildfire Response": {
        "instruction": "What are the best practices for responding to a wildfire?",
        "input": "Rapidly spreading fire in a forested area near residential zones."
    }
}

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FF4B4B;
        margin-bottom: 1rem;
    }
    .response-area {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .stButton button {
        background-color: #FF4B4B;
        color: white;
    }
    .info-box {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🚒 Firefighter Assistant AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='info-box'>This AI assistant has been fine-tuned to provide guidance on firefighting protocols, emergency responses, and safety procedures.</div>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Model Settings</h2>", unsafe_allow_html=True)
    
    model_version = st.radio(
        "Select Model Version",
        ["Final Model", "Best Model (Lowest Loss)"]
    )
    
    device = st.radio(
        "Select Device",
        ["CPU", "CUDA (GPU)"] if torch.cuda.is_available() else ["CPU"]
    )
    
    temperature = st.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1,
                           help="Higher values make output more random, lower values more deterministic")
    
    max_length = st.slider("Max Response Length", min_value=128, max_value=2048, value=512, step=64,
                          help="Maximum number of tokens in the generated response")
    
    st.markdown("<h2 class='sub-header'>Predefined Prompts</h2>", unsafe_allow_html=True)
    selected_prompt = st.selectbox(
        "Choose a predefined prompt",
        list(PREDEFINED_PROMPTS.keys())
    )
    
    if st.button("Use Selected Prompt"):
        st.session_state.instruction = PREDEFINED_PROMPTS[selected_prompt]["instruction"]
        st.session_state.input_text = PREDEFINED_PROMPTS[selected_prompt]["input"]

# Initialize session state for chat history
if 'history' not in st.session_state:
    st.session_state.history = []
if 'instruction' not in st.session_state:
    st.session_state.instruction = ""
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'current_model_path' not in st.session_state:
    st.session_state.current_model_path = None
if 'current_device' not in st.session_state:
    st.session_state.current_device = None

# Function to load model
@st.cache_resource
def load_model(model_path, device_type):
    """Load model and tokenizer with caching"""
    device = "cuda" if device_type == "CUDA (GPU)" and torch.cuda.is_available() else "cpu"
    
    # Log loading info
    st.info(f"Loading model from {model_path} on {device}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    return model, tokenizer, device

# Function to create prompt
def create_prompt(instruction, input_text=""):
    """Create a prompt using the template from training."""
    template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    return template.format(instruction=instruction, input=input_text)

# Function to generate response
def generate_response(model, tokenizer, prompt, device, temperature=0.7, max_length=512):
    """Generate a response from the model."""
    # Create text generation pipeline
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_length=max_length,
        temperature=temperature,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    # Generate response
    generated_text = text_generator(prompt)[0]["generated_text"]
    
    # Extract only the response part
    response_prefix = "### Response:"
    if response_prefix in generated_text:
        response = generated_text.split(response_prefix)[1].strip()
    else:
        response = generated_text
    
    return response

# Main interface
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("<h2 class='sub-header'>Ask the Firefighter Assistant</h2>", unsafe_allow_html=True)
    
    instruction = st.text_area("Instruction", value=st.session_state.instruction, height=100, 
                              placeholder="Enter your question or instruction here...")
    
    input_text = st.text_area("Context (Optional)", value=st.session_state.input_text, height=100,
                             placeholder="Provide additional context if needed...")
    
    if st.button("Generate Response"):
        if not instruction:
            st.error("Please enter an instruction.")
        else:
            # Determine which model to use
            selected_model_path = BEST_MODEL_PATH if model_version == "Best Model (Lowest Loss)" else MODEL_PATH
            selected_device = "cuda" if device == "CUDA (GPU)" and torch.cuda.is_available() else "cpu"
            
            # Check if we need to load a new model
            if (st.session_state.current_model_path != selected_model_path or 
                st.session_state.current_device != selected_device or
                st.session_state.model is None):
                
                with st.spinner("Loading model..."):
                    model, tokenizer, actual_device = load_model(selected_model_path, device)
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.current_model_path = selected_model_path
                    st.session_state.current_device = selected_device
            
            # Generate response
            with st.spinner("Generating response..."):
                prompt = create_prompt(instruction, input_text)
                response = generate_response(
                    st.session_state.model, 
                    st.session_state.tokenizer, 
                    prompt, 
                    "cuda" if st.session_state.current_device == "CUDA (GPU)" else "cpu",
                    temperature,
                    max_length
                )
                
                # Add to history
                st.session_state.history.append({
                    "instruction": instruction,
                    "input": input_text,
                    "response": response
                })
                
                # Clear input fields
                st.session_state.instruction = ""
                st.session_state.input_text = ""

with col2:
    st.markdown("<h2 class='sub-header'>Conversation History</h2>", unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("No conversation history yet. Ask a question to get started!")
    else:
        for i, exchange in enumerate(reversed(st.session_state.history)):
            with st.expander(f"Q: {exchange['instruction'][:50]}...", expanded=(i == 0)):
                st.markdown(f"**Instruction:**\n{exchange['instruction']}")
                if exchange['input']:
                    st.markdown(f"**Context:**\n{exchange['input']}")
                st.markdown("<div class='response-area'>", unsafe_allow_html=True)
                st.markdown(f"**Response:**\n{exchange['response']}")
                st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("**Firefighter Assistant AI** | Developed for French Firefighters | March 2025")
