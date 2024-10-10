#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
from cv_bridge import CvBridge
import sys, select, termios, tty
from pynput import keyboard
from pynput.keyboard import Key
import cv2.aruco as aruco
import numpy as np

# Create a function to capture keyboard input
def get_key():
    tty.setraw(sys.stdin.fileno())
    select.select([sys.stdin], [], [], 0)
    key = sys.stdin.read(1)
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class TelloController(Node):
    def __init__(self):
        super().__init__('tello_controller')

        # Create subscribers and publishers
        self.image_subscriber = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10
        )
        self.cmd_vel_publisher = self.create_publisher(
            Twist, '/cmd_vel', 10
        )

        # CvBridge for converting ROS images to OpenCV
        self.bridge = CvBridge()

        # Set up initial Twist message
        self.twist = Twist()

        # Load the dictionary and parameters for ArUco marker detection
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.aruco_params = aruco.DetectorParameters_create()

         # Camera calibration parameters (Replace these with actual values from camera calibration)
        self.camera_matrix = np.array([[921.170702, 0.000000, 459.904354],
                                       [0.000000, 919.018377, 351.238301],
                                       [0.000000, 0.000000, 1.000000]])
        self.dist_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000]) # Assuming no lens distortion

        # ArUco marker size in meters (set to the actual size of your markers)
        self.marker_size = 0.2  # 20 cm marker

        # Desired distance to maintain (in meters)
        self.desired_distance = 1.0  # 1 meter

        # Keyboard control setup
        self.settings = termios.tcgetattr(sys.stdin)

        # with keyboard.Listener(on_press=self.on_press) as listener:
        #     listener.join()

    def on_press(self, key):
        if key == keyboard.Key.up:
            print('UP')
            self.twist.linear.x = 0.5
            self.twist.angular.z = 0.0
        if key == keyboard.Key.down:
            print('Down')
            self.twist.linear.x = -0.5
            self.twist.angular.z = 0.0
        if key == keyboard.Key.right:
            print('Right')
            self.twist.angular.z = 0.5
            self.twist.linear.x = 0.0
        if key == keyboard.Key.left:
            print('Left')
            self.twist.angular.z = -0.5
            self.twist.linear.x = 0.0
        if key == keyboard.Key.esc:
            print('Down')
            listener.stop()
        self.cmd_vel_publisher.publish(self.twist)

    def image_callback(self, data):
        # Convert the ROS Image message to a format OpenCV can use
        frame = self.bridge.imgmsg_to_cv2(data, "bgr8")

        # ArUco marker detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        # If markers are detected, estimate the pose and control the UAV
        if ids is not None:
            frame = aruco.drawDetectedMarkers(frame, corners, ids)

            # Estimate pose of each marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

            for i in range(len(ids)):
                # Draw the axes for each marker
                aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.1)

                # Translation vector tvecs[i] contains the distance and orientation
                distance = np.linalg.norm(tvecs[i])  # Calculate the distance from the marker

                print(f"Detected ArUco marker ID: {ids[i]}, Distance: {distance:.2f} meters")

                # Control the drone to follow the marker
                self.follow_marker(tvecs[i])

        # Display the image with detected markers and axes
        cv2.imshow("Tello Camera with ArUco Detection", frame)
        cv2.waitKey(1)

    def follow_marker(self, tvec):
        # Extract the x, y, z components of the translation vector (distance to the marker)
        x, y, z = tvec[0]

        # Calculate the error between the current distance and the desired distance
        distance_error = z - self.desired_distance

        # Proportional control constants (these should be tuned for your system)
        Kp_linear = 0.5  # Proportional gain for forward/backward movement
        Kp_angular = 2.0  # Proportional gain for left/right angular movement

        # Control logic
        self.twist.linear.x = Kp_linear * distance_error  # Move forward/backward to maintain the desired distance

        # Adjust angular velocity to keep the marker centered
        self.twist.angular.z = -Kp_angular * x  # Rotate to keep the marker in the center of the frame

        # Publish the velocity command to control the UAV
        self.cmd_vel_publisher.publish(self.twist)
        print(self.twist)

    def destroy_node(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        super().destroy_node()

def main(args=None):
    global settings
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init(args=args)

    controller = TelloController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
