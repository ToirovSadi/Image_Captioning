# this is Streamlit app, for you to test the models

import streamlit as st
import lightning as L
import json
import gdown
import os
from PIL import Image

# Set page config
st.set_page_config(page_title="Image Captioning", layout="wide")

# Header
st.title('Image Captioning')

## About the project
st.markdown("#### This is a simple Image Captioning model, which takes an image as input and generates a caption for the image.")

@st.cache_data(ttl=24*3600, max_entries=1)
def load_info():
    with open("model_info.json", "r") as f:
        info = json.load(f)
    return info
info = load_info()

# Sidebar
st.sidebar.title('Models')
model_name = st.sidebar.selectbox('Select Model', ['Transformer', 'Seq2Seq', 'Seq2Seq_1'])
num_captions = st.sidebar.slider('Select Number of Captions', 1, 10, 3)
with st.sidebar.subheader("Model Info"):
    info_text = ['#### Model Parameters:']
    info_text.append(f"- max_len: {info[model_name.lower()]['max_len']}")
    info_text.append(f"- batch_size: {info[model_name.lower()]['batch_size']}")
    info_text.append(f"- max_epochs: {info[model_name.lower()]['max_epochs']}")
    info_text.append(f"- learning_rate: {info[model_name.lower()]['lr']}")
    info_text.append(f"- bleu_score: {info[model_name.lower()]['bleu_score']}")
    info_text.append("#### Model Specific Parameters:")
    if model_name == 'Transformer':
        info_text.append(f"- hidden_dim: {info[model_name.lower()]['hidden_dim']}")
        info_text.append(f"- num_heads: {info[model_name.lower()]['num_heads']}")
        info_text.append(f"- num_layers: {info[model_name.lower()]['num_layers']}")
        info_text.append(f"- dropout: {info[model_name.lower()]['dropout']}")
        info_text.append(f"- ff_expantion: {info[model_name.lower()]['ff_expantion']}")
    else:
        info_text.append(f"- hidden_dim: {info[model_name.lower()]['hidden_dim']}")
        info_text.append(f"- dropout: {info[model_name.lower()]['dropout']}")
    
    info_text.append(f"#### Source Code: [GitHub]({info[model_name.lower()]['source_code']})")
    info_text.append(f"#### Dataset: {info[model_name.lower()]['dataset']}")
    info_text.append(f"#### Model Link: [GoogleDisk]({info[model_name.lower()]['model_link']})")
    
    st.write('\n'.join(info_text))

## Load the model
@st.cache_resource(ttl=24*3600, max_entries=1)
def load_model(model_name, info):
    from src.models.transformer.model import Transformer
    from src.models.seq2seq.model import Seq2Seq

    link = info['model_link']
    file_name = os.path.join('models',  info['file_name'])
    
    if not os.path.exists(file_name):
        print("link:", link)
        gdown.download(link, file_name, quiet=False, fuzzy=True)
    
    model = (Transformer if model_name == 'Transformer' else Seq2Seq).load_from_checkpoint(file_name, map_location='cpu', device='cpu')
    
    model.to('cpu')
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'device'):
        model.encoder.device = 'cpu'
            
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'device'):
        model.decoder.device = 'cpu'

    return model.eval()


col1, col2 = st.columns(2)
with col1:
    ## Image Upload
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        img = Image.open(uploaded_file)
    
with col2:
    st.write("**Captions for the image will appear here.**")

    if uploaded_file is not None:
        model = load_model(model_name, info[model_name.lower()])
        
        if model_name.lower() == 'transformer':
            from src.models.transformer.predict_model import predict as predict_fn
        else:
            from src.models.seq2seq.predict_model import predict as predict_fn
         
        predictions = predict_fn(model, img, num_captions=num_captions, postprocess=True)
        
        st.write("#### Captions:")
        for i, caption in enumerate(predictions):
            st.write(f"##### {i+1}. {caption}")
