import time
import numpy as np
import h5py
import progressbar
import datetime
import tflearn
from tflearn.layers.core import input_data
import torchvision.models as models
from NetworkSwitch import *
import torch
import torch.nn as nn


class Data():
    def __init__(self, driver_name, rover_name, save_data, framework,
                 filename, network_num, input_shape):
        self.angles = []
        self.images = []
        self.start = time.time()
        self.names = driver_name + '_' + rover_name
        self.save_data = save_data
        self.framework = framework
        self.filename = filename
        self.network_num = network_num
        self.input_shape = input_shape


    def load_network(self):
        if self.framework in ['tf', 'TF']:
            tflearn.config.init_training_mode()
            self.network = input_data(shape=self.input_shape)
            self.network = modelswitch[self.network_num](self.network, drop_prob=1.0)
            self.model = tflearn.DNN(self.network)
            self.model.load(self.filename)
        elif self.framework in ['PT', 'pt']:
            self.model=models.resnet34()
            self.model.fc = nn.Linear(512, 4)
            self.model.cuda()
            self.model.load_state_dict(torch.load(self.filename))
        return self.model

    def add_data(self, image, action):
        self.angles.append(action)
        self.images.append(image)
        print('Collecting Data')
        return

    def save(self):
        if self.save_data in ['y', 'Y', 'yes', 'Yes']:
            print('Saving the Training Data you collected.')
            self.images = np.array(self.images, dtype='uint8')
            self.angles = np.array(self.angles, dtype='float16')

            elapsedTime = int(time.time() - self.start)
            dset_name = str(elapsedTime) + "seconds_" + self.names + ".h5"

            h5f = h5py.File(dset_name, 'w')
            h5f.create_dataset('X', data=self.images)
            h5f.create_dataset('Y', data=self.angles)
            h5f.close()
        return
