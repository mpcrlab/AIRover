from __future__ import print_function
from rover.Data import *
import os, sys
from rover.Pygame_UI import *
from rover import Rover
import time
import numpy as np
#from scipy.misc import imresize


class RoverRun(Rover):
    def __init__(self, fileName, network_name, autonomous, driver, rover, FPS,
                 view, save_data, framework, image_type, normalization,
                 norm_vals, num_out):

        Rover.__init__(self)
        self.FPS = FPS
        self.view = view
        self.speed = 0.5
        self.save_data = save_data
        self.userInterface = Pygame_UI(self.FPS, self.speed)
        self.image = None
        self.quit = False
        self.angle = 0
        self.autonomous = autonomous
        self.image_type = image_type
        self.im_shp = None
        self.act = self.userInterface.action_dict['q']

        if self.autonomous is True:
            if self.image_type in ['color', 'Color']:
                self.im_shp = [None, 130, 320, 3]
            elif self.image_type in ['framestack', 'Framestack']:
                self.im_shp = [None, 130, 320, 3]
                self.framestack = np.zeros([1, 130, 320, self.FPS])
                self.stack = [0, 5, 15]
            elif self.image_type in ['grayscale', 'gray', 'Grayscale']:
                self.im_shp = [None, 130, 320, 1]

        self.d = Data(driver, rover, save_data, framework, fileName,
                      network_name, self.im_shp, normalization, norm_vals,
                      num_out, self.image_type)

        if self.autonomous is True:
            self.d.load_network()

        self.run()


    def run(self):
        while type(self.image) == type(None):
            pass

        while not self.quit:
            s = self.image
            if self.view is True:
                self.userInterface.show_feed(s)

       	    key = self.userInterface.getActiveKey()
            if key == 'z':
                self.quit = True

            if self.autonomous is not True:
                    if key in ['w', 'a', 's', 'd', 'q', ' ']:
                        self.act = self.userInterface.action_dict[key]
                    if self.act[-1] != 9 and self.save_data in ['y', 'Y']:
                        self.d.add_data(s, self.act[-1])
            else:
                s = self.d.normalize(s)
                self.angle = self.d.predict(s)
                self.act = self.userInterface.action_dict[self.angle]

            self.set_wheel_treads(self.act[0], self.act[1])
            self.userInterface.manage_UI()

        # cleanup and stop vehicle
        self.set_wheel_treads(0, 0)
        self.userInterface.cleanup()

        # save training data and close
        self.d.save()
        self.close()
