#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import numpy as np
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

        # Get the x and z coordinates from image2.py
        self.joint1_estimation_x = rospy.Subscriber("/estimation/joint1pos_x", Float64, update_j1_x)
        self.joint1_estimation_z = rospy.Subscriber("/estimation/joint1pos_z", Float64, update_j1_z)
        self.joint23_estimation_x = rospy.Subscriber("/estimation/joint23pos_x", Float64, update_j23_x)
        self.joint23_estimation_z = rospy.Subscriber("/estimation/joint23pos_z", Float64, update_j23_z)
        self.joint4_estimation_x = rospy.Subscriber("/estimation/joint4pos_x", Float64, update_j4_x)
        self.joint4_estimation_z = rospy.Subscriber("/estimation/joint4pos_z", Float64, update_j4_z)
        self.ee_estimation_x = rospy.Subscriber("/estimation/ee_x", Float64, update_ee_x)
        self.ee_estimation_z = rospy.Subscriber("/estimation/ee_z", Float64, update_ee_z)


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
        
    # Factor of 1e-5 avoids division by 0 
    def detect_red(self, image):
        mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
        kernel = np.ones((5, 5), np.uint8)
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

    def pixel2meter(self, image):
        circle1Pos = self.detect_blue(image)
        circle2Pos = self.detect_green(image)
        dist = np.sum((circle1Pos - circle2Pos) ** 2)
        return 3 / np.sqrt(dist)


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

    def estimate_joint1(self, image):
        pass


    # Switch to image2 when green approaches xz plane
    def estimate_joint2(self, image1, image2):
        center_blue = self.detect_blue(image1)
        center_green1 = self.detect_green(image1)
        #center_green2 = self.detect_green(image2)

       
        delta_y = center_blue[0] - center_green1[0]
        delta_x = center_blue[1] - center_green1[1]

        print("Difference x - camera1 ", delta_x)
        print("Difference y - camera1", delta_y)
        
        j_angle = np.arctan2(delta_y, delta_x)          


        print("Blue (y,x) ", center_blue[0], ' ', center_blue[1])
        print("Green 1 (y,x) ", center_green1[0], ' ', center_green1[1])
        print("Difference x ", delta_x)
        print("Difference y ", delta_y)
        print("Angle ", j_angle)
        print('-' * 20)

        return j_angle


    def estimate_joint3(self, image):
        pass


    def estimate_joint4(self, image):
        pass




    ####################### CALLBACKS ##############################

    # Receive data from camera 1, process it, and publish
    def callback1(self, data):
        # Receive the image
        try:
            self.cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
            rospy.Subscriber("camera2/robot/image_raw", Image, self.callback2)
        except CvBridgeError as e:
            print(e)


        # joints given trajectories
    #   self.joint1 = Float64()
    #   self.joint1.data = self.trajectory_joint1(cv_image)
        self.joint2 = Float64()
        self.joint2.data = self.trajectory_joint2(self.cv_image1)
        self.joint3 = Float64()
        self.joint3.data = self.trajectory_joint3(self.cv_image1)
        self.joint4 = Float64()
        self.joint4.data = self.trajectory_joint4(self.cv_image1)

        # joints estimations w/ computer vision
        #self.joint1_estimation = Float64()
        #self.joint1_estimation.data = self.estimate_joint1(cv_image1)  
        self.joint2_estimation = Float64()
        self.joint2_estimation.data = self.estimate_joint2(self.cv_image1, self.cv_image2)
        #self.joint3_estimation = Float64()
        #self.joint3_estimation.data = self.estimate_joint3(cv_image1)
        #self.joint4_estimation = Float64()
        #self.joint4_estimation.data = self.estimate_joint4(cv_image1)

        im2 = cv2.imshow('window2', self.cv_image2)
        im1 = cv2.imshow('window1', self.cv_image1)
        cv2.waitKey(1)
        # Publish the results
        try:
            rate = rospy.Rate(20)
            rate.sleep()
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(self.cv_image1, "bgr8"))
            # Move joints by following given trajectories
            #self.robot_joint1_pub.publish(self.joint1)
            self.robot_joint2_pub.publish(self.joint2)
            self.robot_joint3_pub.publish(self.joint3)
            self.robot_joint4_pub.publish(self.joint4)

            # Publish joint estimation w/ computer vision 
            #self.joint1_estimation_pub.publish(self.joint1_estimation)
            self.joint2_estimation_pub.publish(self.joint2_estimation)
            #self.joint3_estimation_pub.publish(self.joint3_estimation)
            #self.joint4_estimation_pub.publish(self.joint4_estimation)      


        except CvBridgeError as e:
            print(e)


    # Receive data from camera2
    def callback2(self, data):
        try:
            self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
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
