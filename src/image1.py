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

    def detect_red(self, image):
        mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])

    def detect_blue(self, image):
        mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])

    def detect_green(self, image):
        mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return np.array([cx, cy])

    def detect_yellow(self, image):
        mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        M = cv2.moments(mask)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
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
    # TODO: 
    #   - remove orange from image -> this ensures joint spheres can be detected


    def estimate_joint1(self, image):
        pass


    def estimate_joint2(self, image):
        pass


    def estimate_joint3(self, image):
        pass


    def estimate_joint4(self, image):
        pass




    # Recieve data from camera 1, process it, and publish
    def callback1(self, data):
        # Receive the image
        try:
            cv_image1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


        # joints given trajectories
    #   self.joint1 = Float64()
    #   self.joint1.data = self.trajectory_joint1(cv_image)
        self.joint2 = Float64()
        self.joint2.data = self.trajectory_joint2(cv_image1)
        self.joint3 = Float64()
        self.joint3.data = self.trajectory_joint3(cv_image1)
        self.joint4 = Float64()
        self.joint4.data = self.trajectory_joint4(cv_image1)

        # joints estimations w/ computer vision
        #self.joint1_estimation = Float64()
        #self.joint1_estimation.data = self.estimate_joint1(cv_image1)  
        #self.joint2_estimation = Float64()
        #self.joint2_estimation.data = self.estimate_joint2(cv_image1)
        #self.joint3_estimation = Float64()
        #self.joint3_estimation.data = self.estimate_joint3(cv_image1)
        #self.joint4_estimation = Float64()
        #self.joint4_estimation.data = self.estimate_joint4(cv_image1)       




        im1 = cv2.imshow('window1', cv_image1)
        cv2.waitKey(1)
        # Publish the results
        try:
            self.image_pub1.publish(self.bridge.cv2_to_imgmsg(cv_image1, "bgr8"))

            # Move joints by following given trajectories
            #self.robot_joint1_pub.publish(self.joint1)
            self.robot_joint2_pub.publish(self.joint2)
            self.robot_joint3_pub.publish(self.joint3)
            self.robot_joint4_pub.publish(self.joint4)

            # Publish joint estimation w/ computer vision 
            #self.joint1_estimation_pub.publish(self.joint1_estimation)
            #self.joint2_estimation_pub.publish(self.joint2_estimation)
            #self.joint3_estimation_pub.publish(self.joint3_estimation)
            #self.joint4_estimation_pub.publish(self.joint4_estimation)      


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
