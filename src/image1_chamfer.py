#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
import os
from sympy import *
from std_msgs.msg import String
from sensor_msgs.msg import Image, JointState
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
        self.box_coordinates = {'x' : 0, 'y' : 0, 'z' : 0}


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
        self.box_estimation_x = rospy.Subscriber("estimation/box_x/camera2", Float64, self.update_box_x, queue_size=10)
        self.box_estimation_z = rospy.Subscriber("estimation/box_z/camera2", Float64, self.update_box_z, queue_size=10)

        # template for chamfer matching
        self.template_orange_sphere = cv2.imread(os.getcwd() + '/sphere.png', 0)
        self.template_orange_box = cv2.imread(os.getcwd() + '/box.png', 0)
        self.template_blue = cv2.imread(os.getcwd() + '/template_blue.png', 0)
        self.template_yellow = cv2.imread(os.getcwd() + '/template_yellow.png', 0)
        self.template_green = cv2.imread(os.getcwd() + '/template_green.png', 0)
        self.template_red = cv2.imread(os.getcwd() + '/template_red.png', 0)

        # Target detection publishers
        self.target_y_pub = rospy.Publisher("/target_estimation/y", Float64, queue_size=10)
        self.target_z_pub = rospy.Publisher("/target_estimation/z", Float64, queue_size=10)
        self.target_x_pub = rospy.Publisher("/target_estimation/x", Float64, queue_size=10)

        # End effector position publishers
        self.ee_y_pub = rospy.Publisher('ee/estimated_y', Float64, queue_size=10)
        self.ee_x_pub = rospy.Publisher('ee/estimated_x', Float64, queue_size=10)
        self.ee_z_pub = rospy.Publisher('ee/estimated_z', Float64, queue_size=10)

        # Box position publishers
        self.box_x_pub = rospy.Publisher('box/estimated_x', Float64, queue_size=10)
        self.box_y_pub = rospy.Publisher('box/estimated_y', Float64, queue_size=10)
        self.box_z_pub = rospy.Publisher('box/estimated_z', Float64, queue_size=10)

        self.rate = rospy.Rate(20)

        # Time steps
        self.time_previous_step = np.array([rospy.get_time()], dtype='float64')


        # actual angles for closed-loop control
        self.actual_joint_states = {'q1' : 0, 'q2' : 0, 'q3' : 0, 'q4' : 0}
        self.actual_joint_states_sub = rospy.Subscriber('/robot/joint_states', JointState, self.update_actual_joint_states, queue_size=10)

        # actual target position for closed-loop control
        self.actual_target_position = {'x' : 0, 'y' : 0, 'z' : 0}
        self.actual_target_position_x_sub = rospy.Subscriber('/target/x_position_controller/command', Float64, self.update_target_actual_x, queue_size=10)
        self.actual_target_position_y_sub = rospy.Subscriber('/target/y_position_controller/command', Float64, self.update_target_actual_y, queue_size=10)
        self.actual_target_position_z_sub = rospy.Subscriber('/target/z_position_controller/command', Float64, self.update_target_actual_z, queue_size=10)

        # actual box position for null-space control
        self.actual_box_position = {'x' : 0, 'y' : 0, 'z' : 0}
        self.actual_box_position_x_sub = rospy.Subscriber('/target2/x2_position_controller/command', Float64, self.update_box_actual_x, queue_size=10)
        self.actual_box_position_y_sub = rospy.Subscriber('/target2/y2_position_controller/command', Float64, self.update_box_actual_y, queue_size=10)
        self.actual_box_position_z_sub = rospy.Subscriber('/target2/z2_position_controller/command', Float64, self.update_box_actual_z, queue_size=10)


        # Initialize error and derivative of the error
        self.error = np.array([0.0, 0.0, 0.0], dtype='float64')
        self.error_d = np.array([0.0, 0.0, 0.0], dtype='float64')

        # Error and derivate of the error for secondary task
        self.error_secondary = np.array([0.0, 0.0, 0.0, 0.0], dtype='float64')
        self.error_d_secondary = np.array([0.0, 0.0, 0.0, 0.0], dtype='float64')


    # The actual coordinates of the sphere and box are used only for checking correctness of controllers
    # In final report, the estimated values are used
    def update_target_actual_x(self, data):
        self.actual_target_position['x'] = data.data

    def update_target_actual_y(self, data):
        self.actual_target_position['y'] = data.data

    def update_target_actual_z(self, data):
        self.actual_target_position['z'] = data.data

    def update_box_actual_x(self, data):
        self.actual_box_position['x'] = data.data

    def update_box_actual_y(self, data):
        self.actual_box_position['y'] = data.data

    def update_box_actual_z(self, data):
        self.actual_box_position['z'] = data.data


    def update_actual_joint_states(self, data):
        self.actual_joint_states['q1'] = data.position[0]
        self.actual_joint_states['q2'] = data.position[1]
        self.actual_joint_states['q3'] = data.position[2]
        self.actual_joint_states['q4'] = data.position[3]


    def update_j1_x(self,data):
        self.joint1_coordinates['x'] = data.data

    def update_j1_z(self,data):
    	self.joint1_coordinates['z'] = data.data

    def update_j23_x(self,data):
    	self.joint23_coordinates['x'] = data.data

    def update_j23_z(self,data):
    	self.joint23_coordinates['z'] = data.data

    def update_j4_x(self,data):
    	self.joint4_coordinates['x'] = data.data

    def update_j4_z(self,data):
    	self.joint4_coordinates['z'] = data.data

    def update_ee_x(self,data):
   	    self.ee_coordinates['x'] = data.data

    def update_ee_z(self,data):
   	    self.ee_coordinates['z'] = data.data

    def update_target_x(self, data):
        self.target_coordinates['x'] = data.data

    def update_target_z(self, data):
        self.target_coordinates['z'] = data.data

    def update_box_x(self, data):
        self.box_coordinates['x'] = data.data

    def update_box_z(self, data):
        self.box_coordinates['z'] = data.data



    def detect_blue(self, image):
        mask = cv2.inRange(image, (100, 0, 0), (255, 1, 1))
        method = eval('cv2.TM_SQDIFF')
        w, h = self.template_blue.shape[::-1]
        res = cv2.matchTemplate(mask, self.template_blue, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        y = top_left[0] + (w/2)
        z = top_left[1]+ (h/2)

        # cv2.imshow('window1', cv2.rectangle(image, top_left, bottom_right, (0,0,0), 2))

        return np.array([y, z])

    def detect_yellow(self, image):
        mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
        method = eval('cv2.TM_SQDIFF')
        w, h = self.template_yellow.shape[::-1]
        res = cv2.matchTemplate(mask, self.template_yellow, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        y = top_left[0] + (w/2)
        z = top_left[1]+ (h/2)

        # cv2.imshow('window1', cv2.rectangle(image, top_left, bottom_right, (0,0,0), 2))

        return np.array([y, z])

    def detect_green(self, image):
        mask = cv2.inRange(image, (0, 100, 0), (1, 255, 1))
        method = eval('cv2.TM_SQDIFF')
        w, h = self.template_green.shape[::-1]
        res = cv2.matchTemplate(mask, self.template_green, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        y = top_left[0] + (w/2)
        z = top_left[1]+ (h/2)

        # cv2.imshow('window1', cv2.rectangle(image, top_left, bottom_right, (0,0,0), 2))

        return np.array([y, z])

    def detect_red(self, image):
        mask = cv2.inRange(image, (0, 0, 100), (1, 1, 255))
        method = eval('cv2.TM_SQDIFF')
        w, h = self.template_red.shape[::-1]
        res = cv2.matchTemplate(mask, self.template_red, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = min_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        y = top_left[0] + (w/2)
        z = top_left[1]+ (h/2)

        # cv2.imshow('window1', cv2.rectangle(image, top_left, bottom_right, (0,0,0), 2))

        return np.array([y, z])





    def detect_target(self, image, target='sphere'):
        mask = cv2.inRange(image, (5, 50, 100), (10, 80, 150))

        # remove box by finding its contour first and filling it with grey
        edges = cv2.Canny(mask, 30, 200)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

        # Contour approximation to distinguish between sphere and box
        # The contour chosen will be filled with grey, therefore if we want to detect the sphere, fill the box and viceversa
        c = None
        for contour in contours:
            length = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, length*0.04, True)
            if target == 'sphere':
                if len(approx) == 4:
                    c = contour   
            elif target == 'box':
                if len(approx) != 4:
                    c = contour
                    
        image_no_target= image.copy()
        image_no_target = cv2.drawContours(image_no_target, [c], -1, (179, 179, 179), thickness=cv2.FILLED) 

        mask = cv2.inRange(image_no_target, (5, 50, 100), (10, 80, 150))


        # Chamfer matching
        method = eval('cv2.TM_SQDIFF')

        if target == 'sphere':
            w, h = self.template_orange_sphere.shape[::-1]
            res = cv2.matchTemplate(mask, self.template_orange_sphere, method)
        elif target == 'box':
            w, h = self.template_orange_box.shape[::-1]
            res = cv2.matchTemplate(mask, self.template_orange_box, method)


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

        # cv2.imshow('mask camera1', mask)
        # cv2.imshow('window1', cv2.rectangle(image, top_left, bottom_right, (0,0,0), 2))
        
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

    def trajectory_joint1(self, image):
        pass

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
        target = self.detect_target(image, target='sphere')
        self.target_coordinates['y'] = target[0]
        if not (self.target_coordinates['z'] == 0):
            temp = self.target_coordinates['z'] + target[1]
            self.target_coordinates['z'] = temp/2
        else:
            self.target_coordinates['z'] = target[1]

    def estimate_and_update_box(self, image):
        box = self.detect_target(image, target='box')
        self.box_coordinates['y'] = box[0]
        if not(self.box_coordinates['z'] == 0):
            temp = self.box_coordinates['z'] + box[1]
            self.box_coordinates['z'] = temp/2
        else:
            self.box_coordinates['z'] = box[1]


    def transformation_matrix(self, a, alpha, d, theta):
        '''
        Computes transformation matrix A for frame i wrt to frame i-1 by following DH convention
        '''
        
        # Rotation about z
        Rz = Matrix([[cos(theta), -sin(theta), 0, 0], 
                     [sin(theta),  cos(theta), 0, 0], 
                     [0,           0,          1, 0], 
                     [0,           0,          0, 1]])

        # Transition along z
        Tz = Matrix([[1, 0, 0, 0], 
                     [0, 1, 0, 0], 
                     [0, 0, 1, d], 
                     [0, 0, 0, 1]])

        # Transition along x
        Tx = Matrix([[1, 0, 0, a], 
                     [0, 1, 0, 0], 
                     [0, 0, 1, 0], 
                     [0, 0, 0, 1]])

        # Rotation about x
        Rx = Matrix([[1, 0,           0,          0], 
                     [0, cos(alpha), -sin(alpha), 0], 
                     [0, sin(alpha),  cos(alpha), 0], 
                     [0, 0,           0,          1]])


        A = Rz*Tz*Tx*Rx

        return A




    def forward_kinematics_end_effector(self, A_01, A_12, A_23, A_34):
        '''
        Computes Forward Kinematics given transformation matrices
        '''
        FK = A_01*A_12*A_23*A_34
        return FK



    def estimate_jacobian(self, FK, q1, q2, q3, q4):
        theta1, theta2, theta3, theta4 = symbols('theta1 theta2 theta3 theta4')
        end_effector_FK = FK[0:3, 3]
        jacobian = end_effector_FK.jacobian([theta1, theta2, theta3, theta4])
        jac = jacobian.evalf(subs = {theta1:q1, theta2:q2, theta3:q3, theta4:q4})
        return jac


    def control_closed(self, angles, FK, target='sphere'):
        K_p = np.array([[1,0,0],[0,1,0],[0,0,1]])
        K_d = np.array([[0.05,0,0],[0,0.05,0],[0,0,0.05]]) 
        K_i = np.array([[0.15,0,0],[0,0.15,0],[0,0,0.15]])

        c_time = np.array([rospy.get_time()])
        dt = c_time - self.time_previous_step
        self.time_previous_step = c_time

        pos = np.array([(self.ee_coordinates['x'] - self.joint1_coordinates['x']) * self.pixel2meter_ratio, 
                        (self.ee_coordinates['y'] - self.joint1_coordinates['y']) * self.pixel2meter_ratio, 
                        (self.joint1_coordinates['z'] - self.ee_coordinates['z']) * self.pixel2meter_ratio])

        # Depending on whether we want to follow the sphere or the box
        if target == 'sphere':
            pos_d = np.array([self.target_coordinates['x'], 
                              self.target_coordinates['y'], 
                              self.target_coordinates['z']])
        elif target == 'box':
            pos_d = np.array([self.box_coordinates['x'], 
                              self.box_coordinates['y'], 
                              self.box_coordinates['z']])


        self.error_d = ((pos_d - pos) - self.error) / dt
        self.error = pos_d - pos
        q = angles
        jac = np.asarray(self.estimate_jacobian(FK, angles[0], angles[1], angles[2], angles[3])).astype(np.float64)
        jac_inv = np.linalg.pinv(jac)
        d_qd = np.dot(jac_inv, (np.dot(K_d, self.error_d.transpose()) + np.dot(K_p, self.error.transpose()) + np.dot(K_i, (self.error * dt).transpose())))
        q_d = q + (d_qd * dt)
        return q_d


    def control_null_space(self, angles, FK):
        K_p = np.array([[3,0,0],[0,3,0],[0,0,3]])
        K_d = np.array([[0.01,0,0],[0,0.01,0],[0,0,0.01]]) 
        K_i = np.array([[0.15,0,0],[0,0.15,0],[0,0,0.15]])

        # Configuration we want to stay away from
        pos_not_d = self.control_closed(angles, FK, target='box')

        c_time = np.array([rospy.get_time()])
        dt = c_time - self.time_previous_step
        self.time_previous_step = c_time

        pos = np.array([(self.ee_coordinates['x'] - self.joint1_coordinates['x']) * self.pixel2meter_ratio, 
                         (self.ee_coordinates['y'] - self.joint1_coordinates['y']) * self.pixel2meter_ratio, 
                         (self.joint1_coordinates['z'] - self.ee_coordinates['z']) * self.pixel2meter_ratio])


        pos_d = np.array([self.target_coordinates['x'], 
                          self.target_coordinates['y'], 
                          self.target_coordinates['z']])

        

        self.error_d = ((pos_d - pos) - self.error) / dt
        self.error = pos_d - pos

        self.error_d_secondary = (pos_not_d - angles) / dt
        self.error_secondary = pos_not_d - angles

        q = angles
        jac = np.asarray(self.estimate_jacobian(FK, angles[0], angles[1], angles[2], angles[3])).astype(np.float64)
        jac_inv = np.linalg.pinv(jac)
        d_qd = np.dot(jac_inv, (np.dot(K_d, self.error_d.transpose()) + 
                                np.dot(K_p, self.error.transpose()) + 
                                np.dot(K_i, (self.error * dt).transpose())))
        proj_null_space = np.eye(jac_inv.shape[0]) - np.dot(jac_inv, jac)
        d_qd -= np.dot(proj_null_space, self.error_d_secondary)
        q_d = q + (d_qd * dt)
        return q_d






    ########### Estimate angles between points ############
    def estimate_angles_for_j1(self):
        angle_xz = np.arctan2(self.joint23_coordinates['z'] - self.joint1_coordinates['z'], self.joint23_coordinates['x'] - self.joint1_coordinates['x'])
        angle_yz = np.arctan2(self.joint23_coordinates['z'] - self.joint1_coordinates['z'], self.joint23_coordinates['y'] - self.joint1_coordinates['y'])
        return np.array([angle_xz,angle_yz])


      
    def estimate_angles_for_j23(self):
        delta_vertical = self.joint23_coordinates['z'] - self.joint4_coordinates['z']
        delta_horizontal = self.joint4_coordinates['y'] - self.joint23_coordinates['y']
        angle_xz = np.arctan2(delta_vertical, delta_horizontal) - np.pi/2

        delta_horizontal = self.joint4_coordinates['x'] - self.joint23_coordinates['x']
        angle_yz = np.arctan2(delta_vertical, delta_horizontal) - np.pi/2
        return np.array([angle_xz, angle_yz])



    def estimate_angles_for_j4(self):
        temp2 = self.estimate_angles_for_j23()
        if (self.ee_coordinates['z'] > self.joint4_coordinates['z']):
            angle_yz = (-1) * np.arctan2(self.ee_coordinates['z'] - self.joint4_coordinates['z'], self.joint4_coordinates['y'] - self.ee_coordinates['y']) + np.pi/2
        else:
            angle_yz = (-1) * np.arctan2(self.ee_coordinates['z'] - self.joint4_coordinates['z'], self.ee_coordinates['y'] - self.joint4_coordinates['y']) - temp2[1] + np.pi/2 - np.pi
        return angle_yz


    ####################### CALLBACKS ##############################

    # Receive data from camera 1, process it, and publish
    def callback1(self, data):
        # Receive the image
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            #self.orange_binary_mask = self.detect_orange(self.cv_image1)
        except CvBridgeError as e:
            print(e)


        if self.pixel2meter_ratio == 0:
            self.pixel2meter_ratio = self.pixel2meter(self.cv_image1)

        
       

        # joints given trajectories
    #   self.joint1 = Float64()
    #   self.joint1.data = self.trajectory_joint1(cv_image)
        # self.joint2 = Float64()
        # self.joint2.data = self.trajectory_joint2(self.cv_image1)
        # self.joint3 = Float64()
        # self.joint3.data = self.trajectory_joint3(self.cv_image1)
        # self.joint4 = Float64()
        # self.joint4.data = self.trajectory_joint4(self.cv_image1)




        # Get the y and z coordinates from camera 1 and update them accordingly 



        #self.cv_image1_no_orange = self.remove_orange(self.cv_image1)
        self.estimate_and_update_j1(self.cv_image1)
        # self.estimate_and_update_j23(self.cv_image1)
        # self.estimate_and_update_j4(self.cv_image1)
        self.estimate_and_update_ee(self.cv_image1)
        self.estimate_and_update_target(self.cv_image1)
        self.estimate_and_update_box(self.cv_image1)
        

        # joint23_estimation = self.estimate_angles_for_j23()
        # joint4_estimation_x = self.estimate_angles_for_j4()
        
        # self.joint2_estimation = Float64()
        # self.joint2_estimation.data = joint23_estimation[0]
        # self.joint3_estimation = Float64()
        # self.joint3_estimation.data = joint23_estimation[1]
        # self.joint4_estimation = Float64()
        # self.joint4_estimation.data = joint4_estimation_x

        self.target_x = Float64()
        self.target_y = Float64()
        self.target_z = Float64()
        self.target_x.data = self.target_coordinates['x']
        self.target_y.data = self.target_coordinates['y']
        self.target_z.data = self.target_coordinates['z']

        self.ee_x = Float64()
        self.ee_y = Float64()
        self.ee_z = Float64()
        self.ee_x.data = (self.ee_coordinates['x'] - self.joint1_coordinates['x']) * self.pixel2meter_ratio
        self.ee_y.data = (self.ee_coordinates['y'] - self.joint1_coordinates['y']) * self.pixel2meter_ratio
        self.ee_z.data = (self.joint1_coordinates['z'] - self.ee_coordinates['z']) * self.pixel2meter_ratio


        self.box_x = Float64()
        self.box_y = Float64()
        self.box_z = Float64()    
        self.box_x.data = self.box_coordinates['x']
        self.box_y.data = self.box_coordinates['y']
        self.box_z.data = self.box_coordinates['z']

        # print("Joint2 sent ", self.joint2.data)
        # print("Joint2 estimated ", self.joint2_estimation.data)
        # print("Joint3 sent ", self.joint3.data)
        # print("Joint3 estimated ", self.joint3_estimation.data)
        # print("Joint4 sent ", self.joint4.data)
        # print("Joint4 estimated ", self.joint4_estimation.data)
        # print(" ")

        ##################### FORWARD KINEMATICS #####################

    # For 10 settings of the joint angles (theta1, theta2, theta3, theta4) use "rostopic pub" to move the robot (use os.system(command))
    #   Compute end effector position by 
    #       - Computer vision
    #       - Forward Kinematics
    #   output both results and error between them
        theta1, theta2, theta3, theta4 = symbols('theta1 theta2 theta3 theta4')
        A_01 = self.transformation_matrix(0, pi/2, 2.5, (pi/2 + theta1))
        A_12 = self.transformation_matrix(0, pi/2, 0, (pi/2 + theta2))
        A_23 = self.transformation_matrix(3.5, -pi/2, 0, theta3)
        A_34 = self.transformation_matrix(3, 0, 0, theta4)
        FK   = self.forward_kinematics_end_effector(A_01, A_12, A_23, A_34)

        # q1 = 0.5
        # q2 = -0.6
        # q3 = 0.3
        # q4 = -0.3

        # test = FK.evalf(subs = {theta1:q1, theta2:q2, theta3:q3, theta4:q4})
        # ee_pos_FK = np.array(test[0:3, 3])

        # ee_pos_vision = np.array([(self.ee_coordinates['x'] - self.joint1_coordinates['x']) * self.pixel2meter_ratio, 
        #                           (self.ee_coordinates['y'] - self.joint1_coordinates['y']) * self.pixel2meter_ratio, 
        #                           (self.joint1_coordinates['z'] - self.ee_coordinates['z']) * self.pixel2meter_ratio])
        # #print("Error: ", ee_pos_FK - ee_pos_vision)
        # print(ee_pos_FK)
        # print(ee_pos_vision)
        


        ##################### CLOSED-LOOP CONTROL #####################
        #####################  NULL-SPACE CONTROL  #####################

        angles = np.array([self.actual_joint_states['q1'], self.actual_joint_states['q2'], self.actual_joint_states['q3'], self.actual_joint_states['q4']])
        
        # q_d = self.control_closed(angles, FK)
        q_d = self.control_null_space(angles, FK)


        self.joint1 = Float64()
        self.joint1.data = q_d[0]
        self.joint2 = Float64()
        self.joint2.data = q_d[1]
        self.joint3 = Float64()
        self.joint3.data = q_d[2]
        self.joint4 = Float64()
        self.joint4.data = q_d[3]




        
        ######################### SHOW IMAGES #########################

        im1 = cv2.imshow('window1', self.cv_image1)
        #im1_no_orange = cv2.imshow('no_orange camera1', self.cv_image1_no_orange)
        cv2.waitKey(1)


        # Publish the results
        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
            # Move joints by following given trajectories / closed-loop control
            self.robot_joint1_pub.publish(self.joint1)
            self.robot_joint2_pub.publish(self.joint2)
            self.robot_joint3_pub.publish(self.joint3)
            self.robot_joint4_pub.publish(self.joint4)

            # Publish joint estimation w/ computer vision 
            # self.joint1_estimation_pub.publish(self.joint1_estimation)
            # self.joint2_estimation_pub.publish(self.joint2_estimation)
            # self.joint3_estimation_pub.publish(self.joint3_estimation)
            # self.joint4_estimation_pub.publish(self.joint4_estimation)
            self.target_x_pub.publish(self.target_x)
            self.target_y_pub.publish(self.target_y)
            self.target_z_pub.publish(self.target_z)

            self.ee_x_pub.publish(self.ee_x)
            self.ee_y_pub.publish(self.ee_y)
            self.ee_z_pub.publish(self.ee_z)

            self.box_x_pub.publish(self.box_x)
            self.box_y_pub.publish(self.box_y)
            self.box_z_pub.publish(self.box_z)


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
