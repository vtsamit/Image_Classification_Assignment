import streamlit as st
import torch 
from cv2 import cv2 
import torchvision.transforms as transforms 



@st.cache(allow_output_mutation=True)
def load_model():
  model = torch.jit.load('model_scripted.pt')
  
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # IMAGE CLASSIFICATION
         """
         )

file = st.file_uploader("Upload the Image", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):

        size = (32,32)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img=np.asarray(image)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        transform = transforms.Compose([transforms.ToTensor()]) 
        tensor = transform(image)
        p=tensor.unsqueeze(0)
        a=p.to('cuda')
        prediction = model(a)
        q=torch.argmax(prediction)
        class_names1=['airplane', 'automobile','bird','cat', 'deer', 'dog', 'frog','horse','ship','truck']
        ans=class_names1[q]
        return ans
       
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predict = import_and_predict(image, model)
    class_names1=['Airplane', 'Automobile','Bird','Cat', 'Deer', 'Dog', 'Frog','Horse','Ship','Truck']
    string="Image is "+ predict
    st.success(string)
