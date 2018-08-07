from __future__ import print_function
from rover.Data import *
import os, sys
from rover.Pygame_UI import *
from rover import Rover
import time
import numpy as np
from scipy.misc import imresize


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
        self.network = network_name
        self.framework = framework
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
                      network_name, self.im_shp, normalization, norm_vals, num_out)

        if self.autonomous is True:
            self.model = self.d.load_network()

        self.run()


    def run(self):
        start_time = time.time()
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

                if self.framework in ['tf', 'TF']:
                    s = s[None,110:,:,:]

                    if self.image_type in ['grayscale', 'framestack']:
                        s = np.mean(s, 3, keepdims=True)

                    # Framestack
                    if self.image_type in ['framestack']:
                        current = s
                        self.framestack = np.concatenate((current, self.framestack[:, :, :, 1:]), 3)
                        s = self.framestack[:, :, :, self.stack]

                    # predict the correct steering angle from input
                    self.angle = self.model.predict(s)

                elif self.framework in ['pt', 'PT']:
                    s = imresize(s, (224, 224)).transpose((2, 0, 1))[None,...]
                    s = torch.from_numpy(s).float().cuda()
                    self.angle = self.model(s).detach().cpu().numpy()[0, :]

                self.angle = np.argmax(self.angle)
                os.system('clear')
                self.act = self.userInterface.action_dict[self.angle]

            self.set_wheel_treads(self.act[0], self.act[1])
            self.userInterface.manage_UI()

        # print runtime
        elapsed_time = np.round(time.time() - start_time, 2)
        print('This run lasted %.2f seconds'%(elapsed_time))

        # cleanup and stop vehicle
        self.set_wheel_treads(0, 0)
        self.userInterface.cleanup()

        # save training data and close
        self.d.save()
        self.close()
