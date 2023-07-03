# import pandas as pd
import numpy as np
import tensorflow as tf
# from matplotlib import pyplot as plt
# import cv2 as cv
import os
import PIL
from PIL import Image, ImageOps
# import pickle
# import pathlib
# import seaborn as sns
from tensorflow import keras
# from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
# from tensorflow.keras.utils import to_categorical, plot_model
# from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
import streamlit as st

# img_model=VGG16()
# img_model=Model(inputs=img_model.inputs,outputs=img_model.layers[-2].output)

model = keras.models.load_model('./Image_Caption.h5')
# Data_Dir="./Dataset/"
# with open(os.path.join(Data_Dir,"captions.txt"),mode='r') as f:
#     next(f)
#     captions_doc=f.read()
    
# #CREATING A MAPPING OF IMAGES TO CAPTIONS
# mapping={}
# i=0
# for line in captions_doc.split('\n'):
#     #EXTRACT IMAGE ID
#     img_id=(line.split(',')[0]).split('.')[0]
#     caption=(line.split(',')[1:])
#     # print(caption)
#     caption=' '.join(caption)
#     if img_id not in mapping:
#         mapping[img_id]=[]
#     mapping[img_id].append(caption)

# for id,captions in mapping.items():
#     for i in range(len(captions)):
#         caption=captions[i]
#         caption=caption.lower()
#         caption=caption.replace('[^A-Za-z]', '')
#         caption=caption.replace('\s+','')
#         caption='startsent '+" ".join([word for word in caption.split() if len(word)>1])+' endsent'
#         captions[i]=caption

# all_captions=[]
# for key,captions in mapping.items():
#     for i in range(len(captions)):
#         all_captions.append(captions[i])
# all_captions=all_captions[0:-1]

# tokenizer=Tokenizer()
# tokenizer.fit_on_texts(all_captions)
# vocab_size=len(tokenizer.word_index)+1

# max_len_capt=max(len(caption.split()) for caption in all_captions)

# def idx_to_word(i,tokenizer):
#     for word,index in tokenizer.word_index.items():
#         if(i==index):
#             return word
#     return None

# def Predict_Caption(model,image_feature,tokenizer,max_length):
#     text='startsent'
#     for i in range(max_length):
#         seq=tokenizer.texts_to_sequences([text])[0]
#         seq=pad_sequences([seq],maxlen=max_length)
#         y_pred=model.predict([image_feature,seq])
#         y_pred=np.argmax(y_pred)
#         word=idx_to_word(y_pred,tokenizer)
#         if word is None:
#             break
#         text=text+" "+word
#         if(word=="endsent"):
#             break
        
#     return text

st.title("IMAGE TO CAPTION GENERATOR :brain:")

st.header("GENERAL ARCHITECTURE ")
st.image("./model_plot2.png")
test_path="./Testing/"
st.header("TEST IT YOURSELF :smile:")

test_path="./Test/"
upload = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
# if upload is not None:
    # image_bytes = upload.read()
    # st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
    # with open(os.path.join(test_path,"test.jpg"),"wb") as f: 
    #   f.write(upload.getbuffer())    
    # File=os.listdir(test_path)
    # imgpath=""
    # for file in File:
    #     imgpath=os.path.join(test_path,file)
    # test_img=load_img(imgpath,target_size=(224,224))
    # test_img=img_to_array(test_img)
    # test_img=np.expand_dims(test_img,axis=0)
    # test_img=preprocess_input(test_img)
    # feature=img_model.predict(test_img)
    # ans=Predict_Caption(model,feature,tokenizer,max_len_capt)
    # ans=" ".join(ans.split(" ")[1:-1])
    # st.write(ans)
##################################################################
    # image = Image.open(upload)
    # st.image(image, caption="Uploaded Image", use_column_width=True)
    # image = ImageOps.fit(image,(224,224), Image.ANTIALIAS)
    # test_img=img_to_array(image)
    # test_img=np.expand_dims(test_img,axis=0)
    # test_img=preprocess_input(test_img)
    # feature=img_model.predict(test_img)
    # ans=Predict_Caption(model,feature,tokenizer,max_len_capt)
    # ans=" ".join(ans.split(" ")[1:-1])
    # st.write(ans)

# if st.button('Test Result'):
#    st.write(ans)
  
       
