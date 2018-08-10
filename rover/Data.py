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
from scipy.misc import imresize
from skimage.transform import resize

class Data():
    def __init__(self, driver_name, rover_name, save_data, framework,
                 filename, network_name, input_shape, normalization, norm_vals,
                 num_out, image_type):
        self.angles = []
        self.images = []
        self.start = time.time()
        self.names = driver_name + '_' + rover_name
        self.save_data = save_data
        self.framework = framework
        self.filename = filename
        self.network_name =network_name
        self.input_shape = input_shape
        self.normalization = normalization
        self.norm_vals = norm_vals
        self.num_out = num_out
        self.image_type = image_type


    def load_network(self):
        if self.framework in ['tf', 'TF']:
            if self.network_name in ['ResNet34',
                                     'ResNet26',
                                     'ResNeXt34',
                                     'ResNeXt26']:
                tflearn.config.init_training_mode()
            self.network_name = modelswitch[self.network_name]
            self.network = input_data(shape=self.input_shape)
            self.network = self.network_name(self.network,
                                             self.num_out,
                                             drop_prob=1.0)
            self.model = tflearn.DNN(self.network)
            self.model.load(self.filename)

        elif self.framework in ['PT', 'pt']:
            self.network_name = models.__dict__[self.network_name]
            self.model=self.network_name()
            self.model.fc = nn.Linear(512, self.num_out)
            self.model.cuda()
            self.model.load_state_dict(torch.load(self.filename))
            self.model.eval()
        return


    def predict(self, s):
        if self.framework in ['tf', 'TF']:
            s = s[None, 110:, ...]
            if self.image_type in ['grayscale', 'framestack']:
                s = np.mean(s, 3, keepdims=True)
                if self.image_type in ['framestack']:
                    current = s
                    self.framestack = np.concatenate((current,
                                               self.framestack[:, :, :, 1:]), 3)
                    s = self.framestack[:, :, :, self.stack]
            out = self.model.predict(s)

        elif self.framework in ['pt', 'PT']:
            out = resize(s, (224, 224)).transpose((2, 0, 1))[None,...]
            out = torch.from_numpy(out).float().cuda()
            out = self.model(out).detach().cpu().numpy()[0, :]

        return np.argmax(out)


    def normalize(self, x):
        if self.normalization is not None:
            if self.normalization == 'instance_norm':
                x = (x - np.mean(x)) / (np.std(x) + 1e-6)
            elif self.normalization == 'channel_norm':
                for j in range(x.shape[-1]):
                    x[..., j] -= self.norm_vals[j]
        return x


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
