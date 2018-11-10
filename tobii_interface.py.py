import tobii_research as tr
import time
import cv2
import numpy as np
from numpy.random import randint
import csv

class Tobii:
    def __init__(self):
        self.cal_points = 5
        self.ht, self.wd = 1024, 1280  # height and width of the display
        self.point_size = 10 // 2  # number of pixels in a display point
        self.r = list(randint(self.point_size, self.ht-self.point_size, self.cal_points))
        self.c = list(randint(self.point_size, self.wd-self.point_size, self.cal_points))
        self.res = 'calibration_status_failure'
        self.points = 0
        # find the eyetracker
        self.tracker = tr.find_all_eyetrackers()[0]

        while self.res != 'calibration_status_success' or self.points != self.cal_points:
            self.res, self.points = self.calibrate()

    def calibrate(self):
        # instantiate calibration object
        self.cal = tr.ScreenBasedCalibration(self.tracker)
        self.cal.enter_calibration_mode() # enter calibration mode

        for row, col in list(zip(self.r, self.c)):
            img = np.zeros([self.ht, self.wd]) # initialize images with zeros
            img[row-self.point_size:row+self.point_size,
                col-self.point_size:col+self.point_size] = 255.

            row, col = float(row) / self.ht, float(col) / self.wd # normalize the points for calibration

            cv2.namedWindow('test', cv2.WINDOW_NORMAL)
            cv2.imshow('test', img)
            cv2.waitKey(800)

            #if cal.collect_data(col, row) != tr.CALIBRATION_STATUS_SUCCESS:
            self.cal.collect_data(col, row)

        result = self.cal.compute_and_apply()

        print "Compute and apply returned {0} and collected at {1} points.".\
        format(result.status, len(result.calibration_points))

        self.cal.leave_calibration_mode()
        cv2.destroyAllWindows()

        return result.status, len(result.calibration_points)


def gaze_data_callback(gaze_data):
    global global_gaze_data
    global_gaze_data.append([gaze_data['left_gaze_point_on_display_area'],
                             gaze_data['right_gaze_point_on_display_area']])



# tobii = Tobii()
# global global_gaze_data
# global_gaze_data = []
# tobii.tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA,
#                            gaze_data_callback,
#                            as_dictionary=True)
#
# time.sleep(3)
#
#
# tobii.tracker.unsubscribe_from(tr.EYETRACKER_GAZE_DATA, gaze_data_callback)
