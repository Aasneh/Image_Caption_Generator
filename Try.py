import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2 as cv
import os
import PIL
from PIL import Image
import pickle
import pathlib
import streamlit as st

st.title("Image Uploader")    
upload = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
if upload is not None:
    image_bytes = upload.read()
    st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
    with open(os.path.join("./Test/",upload.name),"wb") as f: 
      f.write(upload.getbuffer())         
    st.success("Saved File")