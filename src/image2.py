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
    
    # template for chamfer matching
    self.template_orange_sphere = cv2.imread(os.getcwd() + '/sphere.png', 0)
    self.template_orange_box = cv2.imread(os.getcwd() + '/box.png', 0)
    self.template_blue = cv2.imread(os.getcwd() + '/template_blue.png', 0)
    self.template_yellow = cv2.imread(os.getcwd() + '/template_yellow.png', 0)
    self.template_green = cv2.imread(os.getcwd() + '/template_green.png', 0)
    self.template_red = cv2.imread(os.getcwd() + '/template_red.png', 0)

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
    self.robot_box_x_estimation = rospy.Publisher("estimation/box_x/camera2", Float64, queue_size=10)
    self.robot_box_z_estimation = rospy.Publisher("estimation/box_z/camera2", Float64, queue_size=10)

    self.pixel2meter_ratio = 0

    self.rate = rospy.Rate(20)



  def detect_blue(self, image):
    mask = cv2.inRange(image, (100, 0, 0), (255, 1, 1))
    method = eval('cv2.TM_SQDIFF')
    w, h = self.template_blue.shape[::-1]
    res = cv2.matchTemplate(mask, self.template_blue, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    x = top_left[0] + (w/2)
    z = top_left[1]+ (h/2)
    return np.array([x, z])

  def detect_yellow(self, image):
    mask = cv2.inRange(image, (0, 100, 100), (0, 255, 255))
    method = eval('cv2.TM_SQDIFF')
    w, h = self.template_yellow.shape[::-1]
    res = cv2.matchTemplate(mask, self.template_yellow, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    x = top_left[0] + (w/2)
    z = top_left[1]+ (h/2)
    return np.array([x, z])

  def detect_green(self, image):
    mask = cv2.inRange(image, (0, 100, 0), (1, 255, 1))
    method = eval('cv2.TM_SQDIFF')
    w, h = self.template_green.shape[::-1]
    res = cv2.matchTemplate(mask, self.template_green, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    x = top_left[0] + (w/2)
    z = top_left[1]+ (h/2)
    return np.array([x, z])

  def detect_red(self, image):
    mask = cv2.inRange(image, (0, 0, 100), (1, 1, 255))
    method = eval('cv2.TM_SQDIFF')
    w, h = self.template_red.shape[::-1]
    res = cv2.matchTemplate(mask, self.template_red, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    x = top_left[0] + (w/2)
    z = top_left[1]+ (h/2)
    return np.array([x, z])


  def detect_target(self, image, target='sphere'):
    mask = cv2.inRange(image, (5, 50, 100), (10, 80, 150))
    edges = cv2.Canny(mask, 30, 200)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    #Contour approximation to distinguish between sphere and box
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
    x = top_left[0] + (w/2)
    z = top_left[1] + (h/2)


    # Position of the target must be given wrt to base frame (yellow sphere) in meters
    yellow = self.detect_yellow(image)
    x_yellow = yellow[0]
    z_yellow = yellow[1]
    delta_x = self.pixel2meter_ratio * (x - x_yellow)
    delta_z = self.pixel2meter_ratio * (z_yellow - z)

    # cv2.imshow('detection camera2', cv2.rectangle(image, top_left, bottom_right, (0,0,0), 2))

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

    if self.pixel2meter_ratio == 0:
      self.pixel2meter_ratio = self.pixel2meter(self.cv_image2)

    # Uncomment if you want to save the image
    #cv2.imwrite('image_copy.png', cv_image)
    # im2=cv2.imshow('window2', self.cv_image2)
    # cv2.waitKey(1)

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

    target = self.detect_target(self.cv_image2, target='sphere')
    self.target_x = Float64()
    self.target_x.data = target[0]
    self.target_z = Float64()
    self.target_z.data = target[1]

    # box = self.detect_target(self.cv_image2, target='box')
    # self.box_x = Float64()
    # self.box_x.data = box[0]
    # self.box_z = Float64()
    # self.box_z.data = box[1]


    # Publish the results
    try: 
      self.image_pub2.publish(self.bridge.cv2_to_imgmsg(self.cv_image2, "bgr8"))

      self.robot_joint1_x_estimation.publish(self.joint1x)
      self.robot_joint1_z_estimation.publish(self.joint1z)

      # self.robot_joint23_x_estimation.publish(self.joint23_x)
      # self.robot_joint23_z_estimation.publish(self.joint23_z)

      # self.robot_joint4_x_estimation.publish(self.joint4_x)
      # self.robot_joint4_z_estimation.publish(self.joint4_z)

      self.robot_ee_x_estimation.publish(self.ee_pos_x)
      self.robot_ee_z_estimation.publish(self.ee_pos_z)

      self.robot_target_x_estimation.publish(self.target_x)
      self.robot_target_z_estimation.publish(self.target_z)

      # self.robot_box_x_estimation.publish(self.box_x)
      # self.robot_box_z_estimation.publish(self.box_z)

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


