#!/usr/bin/env python3

import roslib
import sys
import rospy
import cv2
import os
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
    # initialize a publisher to send images from camera2 to a topic named image_topic2
    self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera2/image_raw and use callback function to receive data
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    
    self.template = cv2.imread(os.getcwd() + '/sphere.png', 0)

    self.robot_joint1_x_estimation = rospy.Publisher("/estimation/joint1pos_x", Float64, queue_size=10)
    self.robot_joint23_x_estimation = rospy.Publisher("/estimation/joint23pos_x", Float64, queue_size=10)
    self.robot_joint4_x_estimation = rospy.Publisher("/estimation/joint4pos_x", Float64, queue_size=10)
    self.robot_joint1_z_estimation = rospy.Publisher("/estimation/joint1pos_z", Float64, queue_size=10)
    self.robot_joint23_z_estimation = rospy.Publisher("/estimation/joint23pos_z", Float64, queue_size=10)
    self.robot_joint4_z_estimation = rospy.Publisher("/estimation/joint4pos_z", Float64, queue_size=10)
    self.robot_ee_x_estimation = rospy.Publisher("/estimation/ee_x", Float64, queue_size=10)
    self.robot_ee_z_estimation = rospy.Publisher("/estimation/ee_z", Float64, queue_size=10)
    self.robot_target_x_estimation = rospy.Publisher("estimation/target_x/camera2", Float64, queue_size=10)
    self.robot_target_z_estimation = rospy.Publisher("estimation/target_z/camera2", Float64, queue_size=10)

    self.pixel2meter_ratio = 0

    self.rate = rospy.Rate(20)

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

  def detect_orange(self, image):
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
    x = top_left[0] + (w/2)
    z = top_left[1] + (h/2)
    # Position of the target must be given wrt to base frame (yellow sphere) in meters
    yellow = self.detect_yellow(image)
    x_yellow = yellow[0]
    z_yellow = yellow[1]
    delta_x = self.pixel2meter_ratio * (x - x_yellow)
    delta_z = self.pixel2meter_ratio * (z_yellow - z)
    return np.array([delta_x, delta_z])


  def pixel2meter(self, image):
    yellow = self.detect_yellow(image)
    blue = self.detect_blue(image)
    dist = np.sum((yellow - blue)**2)
    return 2.5 / np.sqrt(dist)




  # Receive data, process it, and publish
  def callback2(self,data):
    # Receive the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)


    self.pixel2meter_ratio = self.pixel2meter(self.cv_image2)

    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)
    im2=cv2.imshow('window2', self.cv_image2)
    cv2.waitKey(1)

    yellow = self.detect_yellow(self.cv_image2)
    self.joint1x = Float64()
    self.joint1x.data = yellow[0]
    self.joint1z = Float64()
    self.joint1z.data = yellow[1]

    blue = self.detect_blue(self.cv_image2)
    self.joint23_x = Float64()
    self.joint23_x.data = blue[0]
    self.joint23_z = Float64()
    self.joint23_z.data = blue[1]

    green = self.detect_green(self.cv_image2)
    self.joint4_x = Float64()
    self.joint4_x.data = green[0]
    self.joint4_z = Float64()
    self.joint4_z.data = green[1]

    red = self.detect_red(self.cv_image2)
    self.ee_pos_x = Float64()
    self.ee_pos_x.data = red[0]
    self.ee_pos_z = Float64()
    self.ee_pos_z = red[1]

    target = self.detect_target(self.cv_image2)
    self.target_x = Float64()
    self.target_x.data = target[0]
    self.target_z = Float64()
    self.target_z.data = target[1]


    # Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))

      self.robot_joint1_x_estimation.publish(self.joint1x)
      self.robot_joint1_z_estimation.publish(self.joint1z)

      self.robot_joint23_x_estimation.publish(self.joint23_x)
      self.robot_joint23_z_estimation.publish(self.joint23_z)

      self.robot_joint4_x_estimation.publish(self.joint4_x)
      self.robot_joint4_z_estimation.publish(self.joint4_z)

      self.robot_ee_x_estimation.publish(self.ee_pos_x)
      self.robot_ee_z_estimation.publish(self.ee_pos_z)

      self.robot_target_x_estimation.publish(self.target_x)
      self.robot_target_z_estimation.publish(self.target_z)

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


