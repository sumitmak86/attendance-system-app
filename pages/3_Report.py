import streamlit as st
from Home import face_rec

st.set_page_config(page_title='Report',layout='wide')
st.subheader('Reporting')

# Retrive logs data
# extract data from redis data list

name = 'attendance:logs'

def load_logs(name,end=-1):
    logs_list = face_rec.r.lrange(name,start=0,end=end)  # extract all the data from database
    return logs_list

# tabs to show the info
tab1, tab2 = st.tabs(['Registered Data', 'Logs'])

with tab1:
    if st.button('Refresh Data'):
        # Retrive the data from Redis database
        with st.spinner('Retriving Data from Redis db'):
            redis_face_db = face_rec.retrive_data(name='academy:register')
            st.dataframe(redis_face_db[['Name','Role']])

with tab2:
    if st.button('Refresh Logs'):
        st.write(load_logs(name=name))

