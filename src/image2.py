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
    # initialize a publisher to send images from camera2 to a topic named image_topic2
    self.image_pub2 = rospy.Publisher("image_topic2",Image, queue_size = 1)
    # initialize a subscriber to recieve messages rom a topic named /robot/camera2/image_raw and use callback function to receive data
    self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw",Image,self.callback2)
    # initialize the bridge between openCV and ROS
    self.bridge = CvBridge()
    

    self.robot_joint1_x_estimation = rospy.Publisher("/estimation/joint1pos_x", Float64, queue_size=10)
    self.robot_joint23_x_estimation = rospy.Publisher("/estimation/joint23pos_x", Float64, queue_size=10)
    self.robot_joint4_x_estimation = rospy.Publisher("/estimation/joint4pos_x", Float64, queue_size=10)
    self.robot_joint1_z_estimation = rospy.Publisher("/estimation/joint1pos_z", Float64, queue_size=10)
    self.robot_joint23_z_estimation = rospy.Publisher("/estimation/joint23pos_z", Float64, queue_size=10)
    self.robot_joint4_z_estimation = rospy.Publisher("/estimation/joint4pos_z", Float64, queue_size=10)
    self.robot_ee_x_estimation = rospy.Publisher("/estimation/ee_x", Float64, queue_size=10)
    self.robot_ee_z_estimation = rospy.Publisher("/estimation/ee_z", Float64, queue_size=10)

    self.rate = rospy.Rate(20)

  def detect_red(self, image):
    # Detect the red pixels
    mask = cv2.inRange(image, (0, 0, 100), (0, 0, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    # Using contours find a centroid
    ret, threshold = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(threshold, 1, 2)
    # If no contours are present, then blob is obstructed and hence try to find the centroid of the next blob down the line
    if len(contours) == 0:
      return self.detect_green(image)
    else:
      M = cv2.moments(contours[0])
      cx = int(M['m10'] / (M['m00'] + 1e-5))
      cy = int(M['m01'] / (M['m00'] + 1e-5))
      return np.array([cx, cy])

  def detect_blue(self, image):
    mask = cv2.inRange(image, (100, 0, 0), (255, 0, 0))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    ret, threshold = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(threshold, 1, 2)
    if len(contours) == 0:
      return self.detect_green(image)
    else:
      M = cv2.moments(contours[0])
      cx = int(M['m10'] / (M['m00'] + 1e-5))
      cy = int(M['m01'] / (M['m00'] + 1e-5))
      return np.array([cx, cy])

  def detect_green(self, image):
    mask = cv2.inRange(image, (0, 100, 0), (0, 255, 0))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)

    ret, threshold = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(threshold, 1, 2)
    if len(contours) == 0:
      return self.detect_blue(image)
    else:
      M = cv2.moments(contours[0])
      cx = int(M['m10'] / (M['m00'] + 1e-5))
      cy = int(M['m01'] / (M['m00'] + 1e-5))
      return np.array([cx, cy])

  def detect_yellow(self, image):
    mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    ret, threshold = cv2.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv2.findContours(threshold, 1, 2)
    if len(contours) == 0:
      return self.detect_blue(image)
    else:
      M = cv2.moments(contours[0])
      cx = int(M['m10'] / (M['m00'] + 1e-5))
      cy = int(M['m01'] / (M['m00'] + 1e-5))
      return np.array([cx, cy])


  # Receive data, process it, and publish
  def callback2(self,data):
    # Receive the image
    try:
      self.cv_image2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
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


