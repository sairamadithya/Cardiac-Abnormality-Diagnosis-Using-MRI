#!/usr/bin/env python
# coding: utf-8

# In[33]:


from __future__ import print_function
import streamlit as st
import numpy as np
import nibabel as nib
import tempfile
import os
import matplotlib.pyplot as plt
import sys
import os
import logging
import six
from radiomics import featureextractor, getFeatureClasses
import radiomics
import SimpleITK as sitk
import pandas as pd
import glob,os
import numpy as np
import cv2
import pickle
from sklearn.preprocessing import *
import re

logger = radiomics.logger
logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file

# Write out all log entries to a file
handler = logging.FileHandler(filename='testLog.txt', mode='w')
formatter = logging.Formatter('%(levelname)s:%(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def display_slice(slice_data):
    fig, ax = plt.subplots()
    ax.imshow(slice_data, cmap='gray')
    ax.axis('off')
    st.pyplot(fig)
def extract_numeric_from_string(s):
    # Regular expression pattern to match numeric values
    pattern = r'[-+]?\d*\.?\d+'
    # Find all numeric values in the string
    matches = re.findall(pattern, s)
    # Convert each match to float and return the first match
    if matches:
        return float(matches[0])
    else:
        return 0  # Return None if no numeric value found
    
st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://t3.ftcdn.net/jpg/05/68/00/22/360_F_568002218_RSy1AA4V7xzMjeEVl6e5PLmOghyT4T50.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
html_temp = """ 
  <div style="background-color:orange ;padding:7px">
  <h2 style="color:black;text-align:center;"><b>Cardiac Abnormality Diagnostic System using MRI<b></h2>
  </div>
  """ 
st.markdown(html_temp,unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload MRI Image", type=["nii", "nii.gz"])

if uploaded_file is not None:
        # Save uploaded file to a temporary directory
        temp_dir = tempfile.TemporaryDirectory()
        temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load MRI data from the temporary file
        mri_data1 = nib.load(temp_file_path)
        mri_data=np.array(mri_data1.dataobj)
        affine = mri_data1.affine
        image = sitk.GetImageFromArray(mri_data)
        image.SetSpacing(affine.diagonal()[::-1])
        depth, height, width = mri_data.shape

        axial_slider = st.slider("Axial Slice", 0, depth - 1, depth // 2)
        coronal_slider = st.slider("Coronal Slice", 0, height - 1, height // 2)
        sagittal_slider = st.slider("Sagittal Slice", 0, width - 1, width // 2)

        # Display slices in the same row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Axial View")
            display_slice(mri_data[axial_slider, :, :])

        with col2:
            st.subheader("Coronal View")
            display_slice(mri_data[:, coronal_slider, :])

        with col3:
            st.subheader("Sagittal View")
            display_slice(mri_data[:, :, sagittal_slider])
uploaded_annotation = st.file_uploader("Upload MRI Annotation", type=["nii", "nii.gz"])
if uploaded_annotation is not None:
    temp_dir1 = tempfile.TemporaryDirectory()
    temp_file_path1 = os.path.join(temp_dir1.name, uploaded_annotation.name)
    with open(temp_file_path1, "wb") as f:
            f.write(uploaded_annotation.getvalue())
    an=nib.load(temp_file_path1)
    mri_an=np.array(an.dataobj)
    affine1 = an.affine
    mask = sitk.GetImageFromArray(mri_an)
    mask.SetSpacing(affine1.diagonal()[::-1])
if st.button('Extract features'):
    featureClasses = radiomics.getFeatureClasses()
    extractor = featureextractor.RadiomicsFeatureExtractor()
    featureVector = extractor.execute(image, mask)
    ff=dict(featureVector)
    keys_to_remove = list(ff.keys())[:22]
    for key in keys_to_remove:
        del ff[key]
    pattern = r"\d+\.\d+"
    for key, value in ff.items():
        if isinstance(value, str):
            numerical_value = re.search(pattern, value)
            if numerical_value:
                ff[key] = float(numerical_value.group())
            else:
                ff[key]= float(np.random.randint(0,100))
        else:
            ff[key]= float(np.random.randint(0,100))
    g6=pd.DataFrame(ff.items())
    b=g6.T
    c=b.iloc[1:]
c1=st.number_input('enter your ED')
d1=st.number_input('enter your ES')
e1=st.number_input('enter your Height')
f1=st.number_input('enter your NbFrame')
g1=st.number_input('enter your Weight')
dd=pd.DataFrame([[1,1,1,1,1],[c1,d1,e1,f1,g1]])
dd.columns=['ED','ES','Height','NbFrame','Weight']
loaded_model_1 = pickle.load(open(r"stage-1 cardiac mri.pkl", 'rb'))
loaded_model_2 = pickle.load(open(r"stage-2 cardiac mri.pkl", 'rb'))
if st.button('Diagnose condition'):
    data=pd.concat([dd,c],axis=1)
    nm=np.array(data.loc[1,:]).reshape(1, -1)
    st.write(nm)
    s1=loaded_model_1.predict(nm)
    if s1==1:
            st.success('Normal')
    else:
            st.error('Abnormality detected. Proceed to find the abnormality.')
            if st.button('Detect abnormality'):
                s2=loaded_model_2.predict(nm)
                if s2==0:
                        st.error('DCM detected')
                elif s2==1:
                        st.error('HCM detected')
                elif s2==2:
                        st.error('MINF detected')
                elif s2==3:
                        st.error('RV detected')
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by Venisha Maheshwari, Kashish Maheshwari and Roshni Katkar<a style='display: block; text-align: center;
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

