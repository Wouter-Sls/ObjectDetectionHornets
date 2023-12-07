import os
import streamlit as st
from ultralytics import YOLO
import shutil
from moviepy.editor import *

os.system('sudo apt-get update')
os.system('sudo apt-get install libgl1-mesa-glx')

#Function to convert avi to mp4
def convert_avi_to_mp4(avi_path, mp4_path):
        # Load the AVI file
        clip = VideoFileClip(avi_path)

        # Write the video to an MP4 file
        clip.write_videofile(mp4_path, codec="libx264", audio_codec="aac")


#Title
st.title("Object Detection Hornets")
st.subheader(':blue[_Created by Wouter Selis & Kieran Cornelissen_] :male-technologist:', divider='rainbow')


st.write('The goal of this project is to make an Artificial Intelligence model to detect Asian hornets on a video using a Yolo library. To detect hornets we trained a YoloV8 model. Below, you can upload a video and set the threshold for detecting hornets. The application will then give back a video with hornet detections.')


#Set your own value for the threshold
threshold = st.slider("Set threshold:", 0.1, 0.5, 0.3, step=0.1, key="myslider")

#Upload video or image
uploaded_file = st.file_uploader("Choose a video file", type=None)

if uploaded_file is not None:

    ############# Save file in video or image folder ###################
    if uploaded_file.type.split("/")[-1] in (["png","jpg", "jpeg"]): 
       
        save_folder = "images"
        path = f"./images/{uploaded_file.name}"
    else:
        save_folder = "videos"
        path = f"./videos/{uploaded_file.name}"

    # Create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Get the filename
    filename = os.path.join(save_folder, uploaded_file.name)

    # Save the file to the specified folder
    with open(filename, 'wb') as f:
        f.write(uploaded_file.getvalue())

    with st.spinner('Please wait while we work our magic.'):
    ###################### Test model with uploaded file ################
        model = YOLO("yoloV8allFotosSmall.pt")
        result=model.predict(conf=threshold,source=path, save=True)

    st.success('Done!')

    #Convert avi file to mp4 to display in streamlit
    if os.path.exists(f"./runs/detect/predict/{uploaded_file.name.split('.')[0]}.avi"):
        avi_file_path = f"./runs/detect/predict/{uploaded_file.name.split('.')[0]}.avi"
        mp4_file_path = f"./runs/detect/predict/{uploaded_file.name}"

        convert_avi_to_mp4(avi_file_path, mp4_file_path)
    
    #Show image/video in streamlit
    if uploaded_file.type.split("/")[-1] in (["png","jpg", "jpeg"]): 
        st.image(f"./runs/detect/predict/{uploaded_file.name}")
    else:
        st.video(f"./runs/detect/predict/{uploaded_file.name}")

    
    #################### Delete images and videos #######################

    if os.path.exists('./runs/detect/predict'):
        shutil.rmtree('./runs/detect/predict')

    if os.path.exists('./images'):
        shutil.rmtree('./images')

    if os.path.exists('./videos'):
        shutil.rmtree('./videos')



    
       






