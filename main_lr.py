# @author: Kien Trinh

import numpy as np
import pandas as pd
import os
import random
from joblib import Parallel, delayed
import multiprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def create_attributes(fileId):
  driverId, tripId = fileId
  # total trip time
  trj = pd.read_csv('drivers/%s/%s.csv' % (driverId, tripId))
  trip_time = trj.shape[0]
  
  # arrays
  vel_xy = trj.diff(-1, axis=0)[0:-1]
  velocity = np.sqrt(vel_xy.x**2 + vel_xy.y**2)
  acc_xy = vel_xy.diff(-1, axis=0)[0:-1]
  accelerations = np.sqrt(acc_xy.x**2 + acc_xy.y**2)
  turn_alpha = np.arctan2(vel_xy.x, vel_xy.y)
  
  # total distance
  total_distance = np.sum(velocity)
  skyway_distance = np.sqrt(sum(np.array(trj[-2:-1])[0]**2))
  
  velocities = list(velocity)
  accelerations = list(accelerations)
  
  standing_time = sum(v < 0.01 for v in velocities)
  standing_speed = sum(a<0.001 for a in accelerations)
  
  avg_speed = np.mean(velocities)
  #avg_speed_up = np.mean([v>0 for v in velocities])
  #avg_speed_down = np.mean([v<0 for v in velocities])
  
  std_speed = np.std(np.absolute(velocities))

  avg_acc = np.mean(accelerations)
  #avg_accel = np.mean([a>0 for a in accelerations])
  #avg_decel = np.mean([a<0 for a in accelerations])
  std_accel = np.std(accelerations)
  
  
  avg_turn = np.mean(list(turn_alpha))
  std_turn = np.std(list(turn_alpha))
  
  attributes = [driverId+'_'+tripId, trip_time, total_distance, skyway_distance, avg_speed, std_speed, 
                avg_acc, std_accel,
                avg_turn, std_turn, standing_time, standing_speed
                ]
  
  return attributes



def create_training_data():
  num_cores = 8

  # getting total number of trips
  list_of_files = [[folder, f.replace('.csv','')] for folder in os.listdir('drivers') if 'DS_Store' not in folder
                 for f in os.listdir('drivers/'+folder) if '.csv' in f]

  raw_data = Parallel( n_jobs=num_cores )(delayed(create_attributes)(i) for i in list_files)
  raw_data = pd.DataFrame(raw_data)
  raw_data.columns = ['driver_trip','trip_time','total_distance','skyway_distance','avg_speed','std_speed',
                      'avg_acc','std_acc','avg_turn','std_turn','standing_time','standing_speed']
  # save to file for later training
  raw_data.to_csv('training_set.csv', index=False)
  return None


def single_driver_prob(driverId, numbNeg, data, list_of_drivers):
  other_drivers = list(set(list_of_drivers)-set(driverId))
  negIds = np.random.choice(other_drivers, size=numbNeg, replace=False)
  # getting random drivers and trips
  negIds = [neg+'_'+str(random.randint(1,200)) for neg in negIds]
  # getting ids of this driver and trips
  posIds = [(driverId+'_'+str(i)).strip() for i in range(1,201)]
  
  train_set = data.loc[posIds+negIds]
  test_set = data.loc[posIds]
  labels = [1]*200+[0]*numbNeg
  
  clf = LogisticRegression()
  clf.fit(train_set, labels)
  # getting probabilitis for positive labels
  probs = clf.predict_proba(test_set)[:,1]

  return posIds, probs


if __name__ == '__main__':
  num_cores = 8
  numbNeg = 200
  
  # creating training data ...
  # create_training_data()
  # or reading training data
  data = pd.DataFrame.from_csv('preprocessed_data.csv')
  
  # getting total number of drivers
  list_of_drivers = [folder for folder in os.listdir('drivers') if 'DS_Store' not in folder]
  
  # getting probs for all drivers
  collections = Parallel( n_jobs=num_cores )(  delayed(single_driver_prob)
        (driverId,numbNeg,data,list_of_drivers) for driverId in list_of_drivers  )
  # reshape the 3D array (2736, 2, 200) to (2736*200, 2)
  new_collections = np.array(collections).swapaxes(1,2).reshape(2736*200,-1).T
  
  # creating sumission file
  submission = pd.DataFrame.from_csv('sampleSubmission.csv')
  submission.loc[new_collections[0]] = new_collections[1]
  submission.to_csv('submission.csv', index=True)
  
  print 'Mission completed!'

