#!/usr/bin/env python3
import rospy
import cv2
import cv2.aruco as aruco
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np

class ArucoDetector:
    def __init__(self):
        rospy.init_node('aruco_detector')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/iris/usb_cam/image_raw', Image, self.image_callback)
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(dictionary, parameters)


        self.camera_matrix = np.array([
            [277.191356, 0.0, 320.5],
            [0.0, 277.191356, 240.5],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        self.marker_length = 0.3



    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        corners, ids, _ = self.detector.detectMarkers(frame)


        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            # for i, marker_corners in enumerate(corners):
            # # marker_corners is a 1x4x2 array
            #     corners_2d = marker_corners[0]  # shape: (4, 2)
            #     center_x = int(corners_2d[:, 0].mean())
            #     center_y = int(corners_2d[:, 1].mean())
            #     marker_id = ids[i][0]

            #     rospy.loginfo(f"Marker ID: {marker_id} Center: ({center_x}, {center_y})")
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

            for i in range(len(ids)):
                # Draw pose axes
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_length * 0.5)

                # Log pose
                rvec = rvecs[i].flatten()
                tvec = tvecs[i].flatten()
                rospy.loginfo(f"ID: {ids[i][0]} | Position: {tvecs} | Rotation (Rodrigues): {rvec}")



        cv2.imshow("Aruco Pose", frame)
        cv2.waitKey(1)

if __name__ == '__main__':
    ArucoDetector()
    rospy.spin()
