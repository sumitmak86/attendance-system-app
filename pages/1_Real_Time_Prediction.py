import streamlit as st
from Home import face_rec

from streamlit_webrtc import webrtc_streamer
import av
import time


st.set_page_config(page_title='Real Time Attendance System')

st.subheader('Real Time Attendance System')


# Retrive the data from Redis database
with st.spinner('Retriving Data from Redis db'):
    redis_face_db = face_rec.retrive_data(name='academy:register')
    st.dataframe(redis_face_db)
    
st.success('Data successfully retrived from Redis db')

# time
waittime = 30  # time in sec
setTime = time.time()
realtimepred = face_rec.RealTimePred() 


# Real Time Prediction

 
#streamlit webrtc

# callback function
def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24") #3d numpy array

    pred_img = realtimepred.face_prediction(img,
                                        redis_face_db,
                                        'facial_features',
                                        ['Name','Role'],
                                        thresh=0.5)

    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waittime:
        realtimepred.saveLogs_Redis()
        setTime = time.time()   # reset time

        print('Save Data to redis Database')


    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")


webrtc_streamer(key="realtimeprediction", video_frame_callback=video_frame_callback,
                rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
                )

