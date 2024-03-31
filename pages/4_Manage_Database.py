import streamlit as st
from Home import face_rec

from streamlit_webrtc import webrtc_streamer
import av
import time


st.set_page_config(page_title='Prediction')
st.subheader('Database')

registration_form = face_rec.RegistrationForm()

with st.spinner('Retriving Data from Redis db'):
    redis_face_db = face_rec.retrive_data(name='academy:register')

# tabs to show the info
tab1, tab2 = st.tabs(['View Database', 'Modify Database'])


with tab1:
    if st.button('Read Data'):
        # Retrive the data from Redis database
        st.dataframe(redis_face_db[['Name','Role']])

with tab2:
    person_name = st.text_input(label='Name',placeholder='First & Last Name')
    role = st.selectbox(label='Select Your Role',options=('Student',
                                                      'Teacher'))

    if st.button('Delete'):
        return_val = registration_form.delete_data_from_db(person_name,role)
        if return_val == True:
            st.success(f"{person_name} Deleted successfully")
        elif return_val == 'name_false':
            st.error('Please enter the valid name')
        elif return_val == 'name_doesnt_exist':
            st.error('Name Does Not Registered')
    

