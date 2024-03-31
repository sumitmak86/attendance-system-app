import numpy as np
import pandas as pd
import cv2

import redis

# insight face 
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# Time
import time
from datetime import datetime
import os

# connect to redis client

# redis-16555.c276.us-east-1-2.ec2.cloud.redislabs.com:16555
# q8vm3QP5vuhGVLPRDOb2d9vrUGlf1Ili

hostname = 'redis-12471.c322.us-east-1-2.ec2.cloud.redislabs.com'
portnumber = 12471
password = '3jnRyIFn8ywmayHSnwbqPPO5aE4NKylt'

r = redis.StrictRedis(host=hostname,
                      port=portnumber,
                      password=password)

# Retrive the data from database
# Retrive data from database
def retrive_data(name):
    retrive_dict = r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series = retrive_series.apply(lambda x:np.frombuffer(x,dtype=np.float32))
    index = retrive_series.index
    index = list(map(lambda x:x.decode(),index))
    retrive_series.index = index
    retrive_df = retrive_series.to_frame().reset_index()
    retrive_df.columns = ['name_role','facial_features']
    retrive_df[['Name','Role']] = retrive_df['name_role'].apply(lambda x:x.split('@')).apply(pd.Series)
    return retrive_df[['Name','Role','facial_features']]




# configure Face Analysis
faceapp = FaceAnalysis(name='buffalo_sc',
                       root='insightface_model',
                       providers=['CPUExecutionProvider'])

faceapp.prepare(ctx_id=0,
                det_size=(640,640),
                det_thresh = 0.5)


# ML Search Algorithm
def ml_search_algorithm(dataframe,feature_column, test_vector, 
                        name_role=['Name','Role'],thresh=0.5):
    """
    Cosine similarity base search algorithm
    """
    
    # Step-1: Take the data Frame
    dataframe = dataframe.copy()
    #Step-2: Index face embedding from the dataframe and convert into array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)
    
    #Step-3: Calculate Cosine Similarity
    similar = pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    #Step-4: Filter the Data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:
        #Step-5: Get the person Name
        data_filter.reset_index(drop=True,inplace=True)
        argmax = data_filter['cosine'].argmax()
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'

    return person_name, person_role


# Real Time Prediction
# Save Logs for every 1 min

class RealTimePred:
    def __init__(self):
        self.logs = dict(name=[],role=[],current_time=[])

    def reset_dict(self):
        self.logs = dict(name=[],role=[],current_time=[])

    def saveLogs_Redis(self):
        # Step 1 Create a log dataframe
        dataframe = pd.DataFrame(self.logs)
        
        # Step 2 drop the duplicate information (distinct name)
        dataframe.drop_duplicates('name',inplace=True)
        
        # Step 3 Push data to redis database
        # encode the data

        name_list = dataframe['name'].tolist()
        role_list = dataframe['role'].tolist()
        ctime_list = dataframe['current_time'].tolist()
        encoded_data = []

        for name,role,ctime in zip(name_list,role_list,ctime_list):
            if name != 'Unknown':
                concat_string = f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)

        if len(encoded_data) > 0:
            r.lpush('attendance:logs',*encoded_data)
        
        self.reset_dict()

    def face_prediction(self,test_image, dataframe,feature_column, test_vector, 
                                name_role=['Name','Role'],thresh=0.5):
            # Step-0 find the current time
        current_time = str(datetime.now())
        # Step-1 take the test image and apply to insight face        
        
        results = faceapp.get(test_image)
        test_copy = test_image.copy()
            
        # Step-2 use for loop and extract each embedding and pass tp ml_search_algorithm
            
        for res in results:
            x1,y1,x2,y2 = res['bbox'].astype(int)
            embeddings = res['embedding']
            person_name, person_role = ml_search_algorithm(dataframe,
                                                            feature_column,
                                                            test_vector=embeddings,
                                                            name_role=name_role,
                                                            thresh=thresh)
            #print(person_name,person_role)
            if person_name == 'Unknown':
                color = (0,0,255)
            else:
                color = (0,255,0)
            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
            
            text_gen = person_name
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)

            # save info log in dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)

        return test_copy


##cRegistration Form
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    
    def reset(self):
        self.sample = 0

    def get_embedding(self,frame):
        # get result from insightface model
        results = faceapp.get(frame,max_num=1)
        embeddings = None
        for res in results:
            self.sample += 1 
            x1,y1,x2,y2 = res['bbox'].astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
            text = f"Sample = {self.sample}"
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)
            #facial feature
            embeddings = res['embedding']
        return frame, embeddings
    
    def delete_data_from_db(self,name,role):
        redis_face_db = retrive_data(name='academy:register')
        list = redis_face_db['Name'].tolist()
        #print(list)
        if name is not None:
            if name.strip() != '':
                if name in list:
                    key1 = f'{name}@{role}'
                    
                else:
                    print("no match")
                    return 'name_doesnt_exist'
            else:
                return 'name_false'
        else:
            return 'name_false'
        #print(key1)
        name='academy:register'
        r.hdel(name,key1)
        return True
    

    def save_data_in_redis_db(self,name,role):

        redis_face_db = retrive_data(name='academy:register')
        list = redis_face_db['Name'].tolist()
        #print(list)
        #print(name)
        # Parameter validation
        
        if name is not None:
            if name.strip() != '':
                if name in list:
                    print("match")
                    return 'name_exist'
                else:
                    key = f'{name}@{role}'
            else:
                return 'name_false'
        else:
            return 'name_false'
        
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'

        # Step 1 Load "face_embedding.txt"
        x_array = np.loadtxt('face_embedding.txt',dtype=np.float32)

        # Step 2 convert to array (proper shape)
        received_sample = int(x_array.size/512)
        x_array = x_array.reshape(received_sample,512)
        x_array = np.asarray(x_array)

        # Step 3 calculate mean embeddings
        x_mean = x_array.mean(axis=0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        # Step 4 Save this into redis database (hashes)
        r.hset(name='academy:register',key=key,value=x_mean_bytes)

        os.remove ('face_embedding.txt')
        self.reset()
    
        return True

    