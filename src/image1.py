#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
import os
from std_msgs.msg import String
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Float64
from cv_bridge import CvBridge, CvBridgeError


class image_converter:

    # Defines publisher and subscriber
    def __init__(self):
        # initialize the node named image_processing
        rospy.init_node('image_processing', anonymous=True)
        # initialize a publisher to send images from camera1 to a topic named image_topic1
        self.image_pub1 = rospy.Publisher("image_topic1", Image, queue_size=1)
        # initialize a subscriber to recieve messages rom a topic named /robot/camera1/image_raw and use callback function to recieve data
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback1)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()

        # Joint position (for now equal to given trajectories)
        self.robot_joint1_pub = rospy.Publisher("/robot/joint1_position_controller/command", Float64, queue_size=10)
        self.robot_joint2_pub = rospy.Publisher("/robot/joint2_position_controller/command", Float64, queue_size=10)
        self.robot_joint3_pub = rospy.Publisher("/robot/joint3_position_controller/command", Float64, queue_size=10)
        self.robot_joint4_pub = rospy.Publisher("/robot/joint4_position_controller/command", Float64, queue_size=10)
     
        # Joint estimation w/ computer vision
        self.joint1_estimation_pub = rospy.Publisher("/estimation/joint1", Float64, queue_size=10)
        self.joint2_estimation_pub = rospy.Publisher("/estimation/joint2", Float64, queue_size=10)
        self.joint3_estimation_pub = rospy.Publisher("/estimation/joint3", Float64, queue_size=10)
        self.joint4_estimation_pub = rospy.Publisher("/estimation/joint4", Float64, queue_size=10)

        # Record the beginning time
        self.start_time = rospy.get_time()

        # Global coordinates
        self.joint1_coordinates = {'x' : 0, 'y' : 0, 'z' : 0}
        self.joint23_coordinates = {'x' : 0, 'y' : 0, 'z' : 0}
        self.joint4_coordinates = {'x' : 0, 'y' : 0, 'z' : 0}
        self.ee_coordinates = {'x' : 0, 'y' : 0, 'z' : 0}
        self.target_coordinates = {'x' : 0, 'y' : 0, 'z' : 0}


        self.pixel2meter_ratio = 0


        # Get the x and z coordinates from image2.py
        self.joint1_estimation_x = rospy.Subscriber("/estimation/joint1pos_x", Float64, self.update_j1_x)
        self.joint1_estimation_z = rospy.Subscriber("/estimation/joint1pos_z", Float64, self.update_j1_z)
        self.joint23_estimation_x = rospy.Subscriber("/estimation/joint23pos_x", Float64, self.update_j23_x)
        self.joint23_estimation_z = rospy.Subscriber("/estimation/joint23pos_z", Float64, self.update_j23_z)
        self.joint4_estimation_x = rospy.Subscriber("/estimation/joint4pos_x", Float64, self.update_j4_x)
        self.joint4_estimation_z = rospy.Subscriber("/estimation/joint4pos_z", Float64, self.update_j4_z)
        self.ee_estimation_x = rospy.Subscriber("/estimation/ee_x", Float64, self.update_ee_x)
        self.ee_estimation_z = rospy.Subscriber("/estimation/ee_z", Float64, self.update_ee_z)
        self.target_estimation_x = rospy.Subscriber("estimation/target_x/camera2", Float64, self.update_target_x, queue_size=10)
        self.target_estimation_z = rospy.Subscriber("estimation/target_z/camera2", Float64, self.update_target_z, queue_size=10)

        # template for chamfer matching
        self.template = cv2.imread(os.getcwd() + '/sphere.png', 0)

        # Target detection publishers
        self.target_y_pub = rospy.Publisher("/target_estimation/y", Float64, queue_size=10)
        self.target_z_pub = rospy.Publisher("/target_estimation/z", Float64, queue_size=10)
        self.target_x_pub = rospy.Publisher("/target_estimation/x", Float64, queue_size=10)

        self.rate = rospy.Rate(20)


    def update_j1_x(self,data):
        self.joint1_coordinates['x'] = data.data

    def update_j1_y(self,data):
        self.joint1_coordinates['y'] = data

    def update_j1_z(self,data):
    	self.joint1_coordinates['z'] = data.data

    def update_j23_x(self,data):
    	self.joint23_coordinates['x'] = data.data

    def update_j23_y(self,data):
        self.joint23_coordinates['y'] = data

    def update_j23_z(self,data):
    	self.joint23_coordinates['z'] = data.data

    def update_j4_x(self,data):
    	self.joint4_coordinates['x'] = data.data

    def update_j4_y(self,data):
        self.joint4_coordinates['y'] = data

    def update_j4_z(self,data):
    	self.joint4_coordinates['z'] = data.data

    def update_ee_x(self,data):
   	    self.ee_coordinates['x'] = data.data

    def update_ee_y(self,data):
        self.ee_coordinates['y'] = data

    def update_ee_z(self,data):
   	    self.ee_coordinates['z'] = data.data

    def update_target_x(self, data):
        self.target_coordinates['x'] = data.data

    def update_target_z(self, data):
        self.target_coordinates['z'] = data.data

        
    # Factor of 1e-5 avoids division by 0 
    def detect_red(self,image):
        # Detect the red pixels 
        mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
        kernel = np.ones((5 ,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / (M['m00'] + 1e-5))
        cy = int(M['m01'] / (M['m00'] + 1e-5))
        return np.array([cx, cy])

    def detect_blue(self, image):
        mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / (M['m00'] + 1e-5))
        cy = int(M['m01'] / (M['m00'] + 1e-5))
        return np.array([cx, cy])

    def detect_green(self, image):
        mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / (M['m00'] + 1e-5))
        cy = int(M['m01'] / (M['m00'] + 1e-5))
        return np.array([cx, cy])

    def detect_yellow(self, image):
        mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / (M['m00'] + 1e-5))
        cy = int(M['m01'] / (M['m00'] + 1e-5))
        return np.array([cx, cy])

    def detect_orange(self, image):
        '''
        Returns binary mask isolating both orange objects
        '''
        mask = cv2.inRange(image, (5, 50, 100), (10, 80, 150))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def detect_target(self, image):
        mask = self.detect_orange(image)
        method = eval('cv2.TM_SQDIFF')
        w, h = self.template.shape[::-1]
        res = cv2.matchTemplate(mask, self.template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        y = top_left[0] + (w/2)
        z = top_left[1]+ (h/2)

        # Position of the target must be given wrt to base frame (yellow sphere) in meters
        yellow = self.detect_yellow(image)
        y_yellow = yellow[0]
        z_yellow = yellow[1]
        delta_y = self.pixel2meter_ratio * (y - y_yellow)
        delta_z = self.pixel2meter_ratio * (z_yellow - z)

        cv2.imshow('window1', cv2.rectangle(image, top_left, bottom_right, 255, 2))
        return np.array([delta_y, delta_z])


    def remove_orange(self, image):
        mask = cv2.inRange(image, (5, 50, 100), (10, 80, 150))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        orange_objects_pixels = cv2.findNonZero(mask)
        orange_object_coordinates = []

        for i in range(orange_objects_pixels.shape[0]):
            pixel = orange_objects_pixels[i]
            for j in range(pixel.shape[0]):
                y = pixel[j][0]
                x = pixel[j][1]
                orange_object_coordinates.append([x, y])

        new_image = image.copy()

        # Using coordinates of orange pixels, check what color is at that position in the original image (original_color)
      
        # 3 CASES: 
        # 1 - original_color != orange -> joint is in front of orange objects -> don't do anything

        # if original_color == orange: 
            # 2 - joints are not overalpping with orange objects -> replace pixel color with grey 
            # 3 - joints are behind orange objects -> replace pixel color with appropriate color (check if joint global coordinates are in +- pixel range)

        for coord in orange_object_coordinates:
            y, x = coord[0], coord[1]
            original_color = image[y, x]
            b = original_color[0]
            g = original_color[1]
            r = original_color[2]

            if ((b >= 5 and b <= 10) and (g >= 50 and g <= 80)):      # orange pixel (either no overalap or joints behind orange objects)
                new_image[y, x] = [179, 179, 179] 
                ee_z = self.ee_coordinates['z']
                ee_y = self.ee_coordinates['y']
                joint4_z = self.joint4_coordinates['z']
                joint4_y = self.joint4_coordinates['y']
                if (ee_y > x-10 and ee_y < x+10 and ee_z > y-10 and ee_z < y+10):
                    new_image[y, x] = [1,1,120]
                elif (joint4_y > x-20 and joint4_y < x+20 and joint4_z > y-20 and joint4_z < y+20):
                    new_image[y, x] = [1, 120, 1]

        return new_image



    def pixel2meter(self,image):
        yellow = self.detect_yellow(image)
        blue = self.detect_blue(image)
        dist = np.sum((yellow - blue)**2)
        return 2.5 / np.sqrt(dist)



    ################### TRAJECTORIES ###################

    def trajectory_joint2(self, image):
        current_time = rospy.get_time() - self.start_time
        joint2 = (np.pi/2)*np.sin((np.pi/15)*current_time)
        return joint2

    def trajectory_joint3(self, image):
        current_time = rospy.get_time() - self.start_time
        joint3 = (np.pi/2)*np.sin((np.pi/18)*current_time)
        return joint3

    def trajectory_joint4(self, image):
        current_time = rospy.get_time() - self.start_time
        joint4 = (np.pi/2)*np.sin((np.pi/20)*current_time)
        return joint4


    ################### JOINT ESTIMATION W/ COMPUTER VISION ###################

    ############## Estimating y and z from camera 1 ################
    def estimate_and_update_j1 (self,image):
        yellow = self.detect_yellow(image)
        self.joint1_coordinates['y'] = yellow[0]
        if not (self.joint1_coordinates['z'] == 0):
            temp = self.joint1_coordinates['z'] + yellow[1]
            self.joint1_coordinates['z'] = temp/2
        else:
            self.joint1_coordinates['z'] = yellow[1]

    def estimate_and_update_j23(self,image):
        blue = self.detect_blue(image)
        self.joint23_coordinates['y'] = blue[0]
        if not (self.joint23_coordinates['z'] == 0):
            temp = self.joint23_coordinates['z'] + blue[1]
            self.joint23_coordinates['z'] = temp/2
        else:
            self.joint23_coordinates['z'] = blue[1]

    def estimate_and_update_j4(self,image):
        green = self.detect_green(image)
        self.joint4_coordinates['y'] = green[0]
        if not (self.joint4_coordinates['z'] == 0):
            temp = self.joint4_coordinates['z'] + green[1]
            self.joint4_coordinates['z'] = temp/2
        else:
            self.joint4_coordinates['z'] = green[1]

    def estimate_and_update_ee(self,image):
        red = self.detect_red(image)
        self.ee_coordinates['y'] = red[0]
        if not (self.ee_coordinates['z'] == 0):
            temp = self.ee_coordinates['z'] + red[1]
            self.ee_coordinates['z'] = temp/2
        else:
            self.ee_coordinates['z'] = red[1]

    def estimate_and_update_target(self, image):
        target = self.detect_target(image)
        self.target_coordinates['y'] = target[0]
        if not (self.target_coordinates['z'] == 0):
            temp = self.target_coordinates['z'] + target[1]
            self.target_coordinates['z'] = temp/2
        else:
            self.target_coordinates['z'] = target[1]

    ########### Estimate angles between points ############
    def estimate_angles_for_j1(self):
        angle_xz = np.arctan2(self.joint23_coordinates['z'] - self.joint1_coordinates['z'], self.joint23_coordinates['x'] - self.joint1_coordinates['x'])
        angle_yz = np.arctan2(self.joint23_coordinates['z'] - self.joint1_coordinates['z'], self.joint23_coordinates['y'] - self.joint1_coordinates['y'])
        return np.array([angle_xz,angle_yz])
      
    def estimate_angles_for_j23(self):
        temp1 = self.estimate_angles_for_j1()
        angle_xz = np.arctan2(self.joint4_coordinates['z'] - self.joint23_coordinates['z'], self.joint4_coordinates['x'] - self.joint23_coordinates['x']) + np.pi/2
        angle_yz = (-1) * np.arctan2(self.joint4_coordinates['z'] - self.joint23_coordinates['z'], self.joint4_coordinates['y'] - self.joint23_coordinates['y'])
        return np.array([angle_xz, angle_yz])

    def estimate_angles_for_j4(self):
        temp2 = self.estimate_angles_for_j23()
        angle_xz = np.arctan2(self.ee_coordinates['z'] - self.joint4_coordinates['z'], self.ee_coordinates['x'] - self.joint4_coordinates['x']) - temp2[0] + np.pi/2
        return angle_xz


    ####################### CALLBACKS ##############################

    # Receive data from camera 1, process it, and publish
    def callback1(self, data):
        # Receive the image
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.orange_binary_mask = self.detect_orange(self.cv_image1)
        except CvBridgeError as e:
            print(e)


        self.pixel2meter_ratio = self.pixel2meter(self.cv_image1)

        
       

        # joints given trajectories
    #   self.joint1 = Float64()
    #   self.joint1.data = self.trajectory_joint1(cv_image)
        self.joint2 = Float64()
        self.joint2.data = self.trajectory_joint2(self.cv_image1)
        self.joint3 = Float64()
        self.joint3.data = self.trajectory_joint3(self.cv_image1)
        self.joint4 = Float64()
        self.joint4.data = self.trajectory_joint4(self.cv_image1)

        # Get the y and z coordinates from camera 1 and update them accordingly 

        #//TODO: decide which method to use for obstruction
        # remove_orange() Vs. contour + centroids method

        #self.cv_image1_no_orange = self.remove_orange(self.cv_image1)
        self.estimate_and_update_j1(self.cv_image1)
        self.estimate_and_update_j23(self.cv_image1)
        self.estimate_and_update_j4(self.cv_image1)
        self.estimate_and_update_ee(self.cv_image1)
        self.estimate_and_update_target(self.cv_image1)
        

        joint23_estimation = self.estimate_angles_for_j23()
        joint4_estimation_x = self.estimate_angles_for_j4()
        
        self.joint2_estimation = Float64()
        self.joint2_estimation.data = joint23_estimation[0]
        self.joint3_estimation = Float64()
        self.joint3_estimation.data = joint23_estimation[1]
        self.joint4_estimation = Float64()
        self.joint4_estimation.data = joint4_estimation_x

        self.target_x = Float64()
        self.target_y = Float64()
        self.target_z = Float64()
        self.target_x.data = self.target_coordinates['x']
        self.target_y.data = self.target_coordinates['y']
        self.target_z.data = self.target_coordinates['z']


#########################################################
#####################################################
#######################################
    

        im1 = cv2.imshow('window1', self.cv_image1)
        orange_binary_image = cv2.imshow('orange mask', self.orange_binary_mask)
        #im1_no_orange = cv2.imshow('no_orange camera1', self.cv_image1_no_orange)
        cv2.waitKey(1)
        # Publish the results
        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
            # Move joints by following given trajectories
            #self.robot_joint1_pub.publish(self.joint1)
            self.robot_joint2_pub.publish(self.joint2)
            self.robot_joint3_pub.publish(self.joint3)
            self.robot_joint4_pub.publish(self.joint4)

            # Publish joint estimation w/ computer vision 
            #self.joint1_estimation_pub.publish(self.joint1_estimation)
            self.joint2_estimation_pub.publish(self.joint2_estimation)
            self.joint3_estimation_pub.publish(self.joint3_estimation)
            self.joint4_estimation_pub.publish(self.joint4_estimation)
            self.target_x_pub.publish(self.target_x)
            self.target_y_pub.publish(self.target_y)
            self.target_z_pub.publish(self.target_z)


        except CvBridgeError as e:
            print(e)



# call the class
def main(args):
    ic = image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()


# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
