# Image Caption Generator
### BASIC OVERVIEW :
This Project is an attempt to generate **meaningful and appropriate textual captions** for images given as input. Generating captions, short descriptions and thumbnails for an image may be required and can be a time consuming task. This Project is an effort to effectively resolve such problems.<br>

### :floppy_disk:  DATA COLLECTION AND PREPROCESSING:
**DATASET LINK :** <https://www.kaggle.com/datasets/adityajn105/flickr8k> Flickr8k Dataset used .<br> 
It has **8091 images with about 4-5 meaningful textual captions for each image** .<br>
For Image Feature Extraction I have used a popular technique in Machine Learning called *Transfer Learning* .It uses a pretrained model to extract certain features of the input.<br>
We shall be using the popular **VGG16 Model** to extract features from the input images .<br>
#### VGG MODEL:
It is a 16 layer Deep Convolutional Neural Network . It has 13 **CNN** layers and 3 **Dense Fully Connected Layers** .It was pretrained to classify **1000 images**.
It has an accuracy of about **0.92**. **Relu** is used as activation function and **softmax** in the last layer as the output class is large.Input Image must be of size **(224,224)** and we will use the output of the **2nd** last layer to get the final features of the image.<br>

All the images are converted into arrays of size **(224,224,3)**(RGB Images). Each **2-D matrix** handles individually the components of **red, blue, green** pixels.
Each image's features are extracted and stored using the **VGG model** .<br>



link :https://imagecaptiongenerator.streamlit.app/
