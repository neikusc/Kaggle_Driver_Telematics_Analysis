import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import logging
from neon.backends import gen_backend
from neon.layers import FCLayer, DataLayer, CostLayer
from neon.models import MLP
from neon.transforms import RectLin, Logistic, CrossEntropy
from neon.datasets.dataset import Dataset
from neon.experiments import FitExperiment

#logging.basicConfig(level=20)
#logger = logging.getLogger()


class Axa(Dataset):
  def load(self, **kwargs):
    dtype = np.float32
    
    data = pd.DataFrame.from_csv('preprocessed_data.csv')
    
    driverId = '36'
    numbNeg = 400
    
    list_of_drivers = [folder for folder in os.listdir('drivers')
                       if 'DS_Store' not in folder]
    other_drivers = list(set(list_of_drivers)-set(driverId))
    negIds = np.random.choice(other_drivers, size=numbNeg, replace=False)
    negIds = [neg+'_'+str(random.randint(1,200)) for neg in negIds]
    posIds = [driverId+'_'+str(i) for i in range(1,201)]
    train = data.loc[posIds+negIds]
    test = data.loc[posIds]
                       
    scale = StandardScaler()
    train = scale.fit_transform(train)
    test = scale.transform(test)
                       
    labels = [1]*len(posIds)+[0]*len(negIds)
                       
    # The first column contains labels.
    self.inputs['train'] = train
    self.inputs['test'] = test
    self.targets['train'] = np.empty((train.shape[0], 1), dtype=dtype)
    self.targets['test'] = np.zeros((test.shape[0], 1), dtype=dtype)
                       
    self.targets['train'][:, 0] = labels
    self.format()


def create_model(nin):
  layers = []
  layers.append(DataLayer(nout=nin))
  layers.append(FCLayer(nout=40, activation=RectLin()))
  layers.append(FCLayer(nout=1, activation=Logistic()))
  layers.append(CostLayer(cost=CrossEntropy()))
  model = MLP(num_epochs=10, batch_size=100, layers=layers)
  return model


def run():
  model = create_model(nin=11)
  backend = gen_backend(rng_seed=0)
  dataset = Axa()
  
  experiment = FitExperiment(model=model,
                             backend=backend,
                             dataset=dataset)
  experiment.run()
  outputs, targets = model.predict_fullset(dataset, 'test')
  print outputs.asnumpyarray()

if __name__ == '__main__':
  run()