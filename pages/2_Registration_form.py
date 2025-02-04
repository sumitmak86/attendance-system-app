import streamlit as st
from Home import face_rec
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av 


st.set_page_config(page_title='Prediction')
st.subheader('Registration Form')

## Init Registration Form
registration_form = face_rec.RegistrationForm()

# Step 1 collect person name and role
# Form

person_name = st.text_input(label='Name',placeholder='First & Last Name')
role = st.selectbox(label='Select Your Role',options=('Student',
                                                      'Teacher'))


# Step 2 collect facial embedding
def video_callback_func(frame):
    img = frame.to_ndarray(format='bgr24')  # 3d array bgr

    
    reg_img, embedding = registration_form.get_embedding(img)

    # 2 step process
    # 1st step save data to local machine in txt format
    if embedding is not None:
        with open('face_embedding.txt','ab') as f:
            np.savetxt(f,embedding)
    return av.VideoFrame.from_ndarray(reg_img,format='bgr24')

webrtc_streamer(key='registration',video_frame_callback=video_callback_func,
                rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
                )


# Step 3 Save data in redis database
if st.button('Submit'):
    return_val = registration_form.save_data_in_redis_db(person_name,role)
    if return_val == True:
        st.success(f"{person_name} registered successfully")
    elif return_val == 'name_false':
        st.error('Please enter the valid name')
    elif return_val == 'name_exist':
        st.error('Name already registered')
        st.button('Update','Discard')
    elif return_val == 'file_false':
        st.error('Facial Video Input is not captured') 