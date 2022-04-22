import io
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import streamlit as st
from PIL import Image


model_url = "plant-disease-resnet.pth"
batch_size = 1

map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

# set the model to inference mode
torch_model.eval()


file_up = st.file_uploader("Upload an image", type="jpg")