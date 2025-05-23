#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import tf2_ros
import os
from alterego_msgs.msg import MarkerInfo
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TransformStamped
from filterpy.kalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool, String, Float32

import json
class KalmanPoseFilter:
    def __init__(self, dt=1.0/20.0):  # Aggiornato per 30 Hz
        # Create a Kalman filter for position and orientation
        # State: [x, y, z, vx, vy, vz, qx, qy, qz, qw]
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        
        # State transition matrix (F)
        self.kf.F = np.eye(10)
        self.kf.F[0, 3] = dt  # x += vx * dt
        self.kf.F[1, 4] = dt  # y += vy * dt
        self.kf.F[2, 5] = dt  # z += vz * dt
        
        # Measurement matrix (H)
        self.kf.H = np.zeros((7, 10))
        self.kf.H[0, 0] = 1  # x
        self.kf.H[1, 1] = 1  # y
        self.kf.H[2, 2] = 1  # z
        self.kf.H[3, 6] = 1  # qx
        self.kf.H[4, 7] = 1  # qy
        self.kf.H[5, 8] = 1  # qz
        self.kf.H[6, 9] = 1  # qw
        
        # Aumentare SIGNIFICATIVAMENTE la covarianza del processo per più reattività
        self.kf.Q = np.eye(10) * 0.3  # Aumentato da 0.1 a 0.3
        self.kf.Q[3:6, 3:6] *= 0.5  # Aumentato da 0.2 a 0.5 per velocità
        self.kf.Q[6:, 6:] *= 0.1     # Aumentato da 0.01 a 0.1 per rotazione

        # Ridurre l'influenza delle misurazioni rumorose
        self.kf.R = np.eye(7) * 0.01  # Ridotto da 0.05 a 0.01 per dare più peso alle misurazioni
        self.kf.R[3:, 3:] *= 0.002    # Ridotto da 0.005 a 0.002 per quaternioni
        

        # Aumentare l'adattabilità iniziale
        self.kf.P = np.eye(10) * 10.0  # Più incertezza iniziale
        self.kf.P[6:, 6:] *= 0.5
        
        # Initialize state (x)
        self.kf.x = np.zeros(10)
        self.kf.x[9] = 1.0  # Initial quaternion w=1 (identity rotation)
        
        # Buffer delle misurazioni recenti per l'outlier rejection
        self.position_buffer = []
        self.quaternion_buffer = []
        self.buffer_size = 3  # Mantieni le ultime 5 misurazioni
        
        self.initialized = False
        self.last_update_time = rospy.Time.now()

        
    def normalize_quaternion(self):
        """Normalize the quaternion in the state vector"""
        q = self.kf.x[6:10]
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            self.kf.x[6:10] = q / q_norm
    
    def is_outlier(self, position, quaternion):
        if len(self.position_buffer) < 3:
            return False
        
        # Calcola varianza delle posizioni
        pos_variance = np.var(self.position_buffer, axis=0)
        
        # Soglie dinamiche MOLTO più permissive per evitare falsi positivi
        pos_threshold = np.max(pos_variance) * 5.0  # Aumentato da 5 a 10 volte la varianza
        
        # Verifica outlier con soglie più elastiche
        pos_distance = np.linalg.norm(position - np.mean(self.position_buffer, axis=0))
        
        return pos_distance > pos_threshold
    
    def update_buffers(self, position, quaternion):
        """Aggiorna i buffer con le nuove misurazioni"""
        self.position_buffer.append(position)
        self.quaternion_buffer.append(quaternion)
        
        # Mantieni solo gli ultimi buffer_size elementi
        if len(self.position_buffer) > self.buffer_size:
            self.position_buffer.pop(0)
            self.quaternion_buffer.pop(0)
            
    def update(self, position, quaternion):
        """Update the filter with a new measurement"""

        
        current_time = rospy.Time.now()
        dt = (current_time - self.last_update_time).to_sec()
        
        # Handle first measurement
        if not self.initialized:
            self.kf.x[0] = position[0]  # x
            self.kf.x[1] = position[1]  # y
            self.kf.x[2] = position[2]  # z
            self.kf.x[6] = quaternion[0]  # qx
            self.kf.x[7] = quaternion[1]  # qy
            self.kf.x[8] = quaternion[2]  # qz
            self.kf.x[9] = quaternion[3]  # qw
            self.initialized = True
            self.last_update_time = current_time
            self.update_buffers(position, quaternion)
            return self.kf.x[0:3], self.kf.x[6:10]
            
        # Update dt in state transition matrix
        if dt > 0.001:  # Avoid division by zero
            self.kf.F[0, 3] = dt
            self.kf.F[1, 4] = dt
            self.kf.F[2, 5] = dt
            
        # Ensure quaternion consistency (avoid sign flips)
        if np.dot(quaternion, self.kf.x[6:10]) < 0:
            quaternion = -np.array(quaternion)
        
        # Controlla se questa misurazione è un outlier
        if self.is_outlier(position, quaternion):
            rospy.logdebug("Outlier detected, using prediction only")
            self.kf.predict()
        else:
            # Prediction step
            self.kf.predict()
            
            # Update step
            z = np.array([position[0], position[1], position[2], 
                         quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
            self.kf.update(z)
            
            # Aggiorna i buffer con questa nuova misurazione valida
            self.update_buffers(position, quaternion)
        
        # Normalize quaternion
        self.normalize_quaternion()
        
        # Applicazione di filtro passa-basso aggiuntivo (media mobile)
        self.alpha = 0.9  # Fattore di smoothing (0.0-1.0), più alto = meno smoothing # Da 0.8 a 0.5, più basso = più reattivo
       # Applica smoothing solo se abbiamo abbastanza campioni nel buffer
        if len(self.position_buffer) > 1:
            # Calcola la media pesata tra lo stato corrente e la media delle misurazioni recenti
            weighted_pos = self.alpha * position + (1-self.alpha) * np.mean(self.position_buffer, axis=0)
            
            # Applica il risultato filtrato allo stato - CAMBIATO per usare la misurazione diretta
            self.kf.x[0:3] = weighted_pos
                
        self.last_update_time = current_time
        return self.kf.x[0:3], self.kf.x[6:10]
        
    def get_current_estimate(self):
        """Get the current state estimate without updating"""
        if not self.initialized:
            return np.zeros(3), np.array([0, 0, 0, 1])
            
        # Predict the current state
        dt = (rospy.Time.now() - self.last_update_time).to_sec()
        if dt > 0.001:
            self.kf.F[0, 3] = dt
            self.kf.F[1, 4] = dt
            self.kf.F[2, 5] = dt
            self.kf.predict()
            self.last_update_time = rospy.Time.now()
            self.normalize_quaternion()
            
        return self.kf.x[0:3], self.kf.x[6:10]
class ArucoDetector:
    def __init__(self):
        """Initialize the ArUco marker detector node."""
        rospy.init_node('aruco_detector', anonymous=True)
        
        # Get the robot name from the environment variable
        robot_name = os.getenv('ROBOT_NAME', 'robot_alterego3')
        
        # Initialize Kalman filter
        self.pose_filter = KalmanPoseFilter()
        
        # ArUco dictionary types
        self.dictionary_types = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
            "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
        }
        
        # Configurable parameters
        self.aruco_dictionary_name = rospy.get_param('~aruco_dictionary', 'DICT_6X6_250')
        self.display_detection = rospy.get_param('~display_detection', False)
        self.camera_id = rospy.get_param('~camera_id', 0)
        self.floating_offset = rospy.get_param('~floating_offset', -1.25)
        
        # Subscribe to topic for dynamically changing the floating offset
        rospy.Subscriber(f'/{robot_name}/floating_offset', Float32, self.update_floating_offset)
        # Get dictionary type from configuration
        aruco_dict_type = self.dictionary_types.get(self.aruco_dictionary_name, cv2.aruco.DICT_6X6_250)
        # OpenCV 5.x API for ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict)
        
        # Bridge for OpenCV/ROS conversion
        self.bridge = CvBridge()
        
        # Publishers
        self.detection_pub = rospy.Publisher(f'/{robot_name}/aruco_detected', Bool, queue_size=10)
        self.marker_info_pub = rospy.Publisher(f'/{robot_name}/aruco_marker_info', MarkerInfo, queue_size=10)
        self.detections_pub = rospy.Publisher(f'/{robot_name}/detections', String, queue_size=10)
        self.floating_ready_pub = rospy.Publisher(f'/{robot_name}/floating_marker_ready', Bool, queue_size=10)

        # Parameters for marker tracking
        self.target_marker_id = rospy.get_param('~target_marker_id', 23)
        self.min_marker_size = rospy.get_param('~min_marker_size', 30)  # pixels
        
        # Open camera
        # Tentativi di apertura della camera
        max_attempts = 10
        attempt = 0
        self.cap = None
        
        while attempt < max_attempts:
            self.cap = cv2.VideoCapture(self.camera_id)
            if self.cap.isOpened():
                rospy.loginfo(f"Camera {self.camera_id} aperta con successo al tentativo {attempt + 1}")
                break
            else:
                rospy.logwarn(f"Tentativo {attempt + 1}: impossibile aprire la camera {self.camera_id}")
                self.cap.release()
                rospy.sleep(1)  # Aspetta un secondo prima di riprovare
                attempt += 1
                
        if not self.cap or not self.cap.isOpened():
            rospy.logerr(f"Impossibile aprire la camera {self.camera_id} dopo {max_attempts} tentativi")
            return
        
        # Set higher resolution for better detection
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Original camera matrix for ZED Mini (right camera, 720p)
        self.camera_matrix = np.array([
            [690.76497363, 0.0, 631.08160981],    # fx, 0, cx
            [0.0, 693.70863262, 375.37565261],    # 0, fy, cy
            [0.0, 0.0, 1.0]             # 0, 0, 1
        ])
        
        self.dist_coeffs = np.array([[-0.1355764,  -0.08091611, 0.0051946, -0.00297882, 0.12494643]])
        
        # Set marker size in meters
        self.marker_size = rospy.get_param('~marker_size', 0.20)  # 9cm default
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Frame names
        self.base_frame = "camera_right"
        self.marker_frame = "aruco_marker"
        
        # Variables to track marker detection
        self.marker_visible = False
        self.marker_lost_time = None
        self.marker_redetection_timeout = rospy.Duration(1.0)  # 1 second timeout
        
        rospy.loginfo(f"ArUco detection node started with Kalman filtering. Dictionary: {self.aruco_dictionary_name}")


        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.map_frame = "map"  # Or whatever your map frame is called

        self.should_process = False

    def update_floating_offset(self, msg):
        """
        Callback to update the floating offset based on incoming messages.
        
        Args:
            msg: Float32 message containing the new offset value
        """
        self.floating_offset = msg.data
        self.should_process = True
        rospy.loginfo(f"Updated floating offset to: {self.floating_offset} meters")


    def run(self):
        """Main detection loop running at 15Hz."""
        rate = rospy.Rate(20)  # 15Hz
        
        while not rospy.is_shutdown():
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("Cannot receive frame from camera")
                continue
                
            # Take only right half of the frame (right image from ZED)
            height, width = frame.shape[:2]
            mid_width = width // 2
            frame = frame[:, 0:mid_width]
            
            # Create copy for visualization
            display_frame = frame.copy()
                
            # Convert to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect ArUco markers
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
            
            # Debug: Show rejected markers
            if rejected and self.display_detection:
                cv2.aruco.drawDetectedMarkers(display_frame, rejected, borderColor=(100, 0, 240))
            
            # Check if target marker is detected
            target_detected = False

            # Dizionario per le informazioni sui marker rilevati (simile a face_recognition)
            marker_detections = {}

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
                
                # Process each detected marker
                for i, corner in enumerate(corners):
                    marker_id = ids[i][0]
                    if marker_id == self.target_marker_id:
                        target_detected = True
                        self.marker_visible = True
                        self.marker_lost_time = None
                        
                        c = corner[0]
                        center = (int(c[:, 0].mean()), int(c[:, 1].mean()))
                        
                        # Calculate marker width and center position
                        center_x = c[:, 0].mean()
                        frame_center_x = frame.shape[1] / 2
                        error_x = center_x - frame_center_x
                        
                                                # Calcola l'area del marker nell'immagine
                        marker_area = cv2.contourArea(c.astype(np.float32))
                        
                        # Aggiunge le informazioni del marker al dizionario
                        marker_detections[f"marker_{marker_id}"] = {
                            "corners": c.tolist(),
                            "center": center,
                            "area": float(marker_area),
                            "center_error": float(error_x)
                        }

                        
                        # Estimate pose for the marker
                        try:
                            objPoints = np.array([[-self.marker_size/2, self.marker_size/2, 0],
                                               [self.marker_size/2, self.marker_size/2, 0],
                                               [self.marker_size/2, -self.marker_size/2, 0],
                                               [-self.marker_size/2, -self.marker_size/2, 0]], dtype=np.float32)
                            
                            success, rvec, tvec = cv2.solvePnP(objPoints, 
                                                             corner[0], 
                                                             self.camera_matrix, 
                                                             self.dist_coeffs)
                            
                            if success:
                                # Convert tvec to regular float values
                                tx, ty, tz = float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])
                                
                                # Convert rotation vector to quaternion
                                rot_matrix, _ = cv2.Rodrigues(rvec)
                                quat = self.rotation_matrix_to_quaternion(rot_matrix)
                                
                                # Apply Kalman filtering to position and orientation
                                position = np.array([tx, ty, tz])
                                quaternion = np.array([quat[0], quat[1], quat[2], quat[3]])
                                
                                filtered_position, filtered_quaternion = self.pose_filter.update(position, quaternion)
                                filtered_quaternion = self.publish_map_aligned_marker_frame(marker_id, filtered_position)
                                # filtered_position = position
                                # filtered_quaternion = quaternion
                                filtered_position, filtered_quaternion = self.rotate_marker_axes(filtered_position, filtered_quaternion)
                                
                                # Create MarkerInfo message with filtered values
                                marker_info = MarkerInfo()
                                marker_info.id = marker_id
                                marker_info.center_error = error_x
                                
                                # Fill pose information with filtered values
                                marker_info.pose.position.x = filtered_position[0]
                                marker_info.pose.position.y = filtered_position[1]
                                marker_info.pose.position.z = filtered_position[2]
                                marker_info.pose.orientation.x = filtered_quaternion[0]
                                marker_info.pose.orientation.y = filtered_quaternion[1]
                                marker_info.pose.orientation.z = filtered_quaternion[2]
                                marker_info.pose.orientation.w = filtered_quaternion[3]
                                
                                # Update TF transformation
                                transform = TransformStamped()
                                transform.header.stamp = rospy.Time.now()
                                transform.header.frame_id = self.base_frame
                                transform.child_frame_id = f"{self.marker_frame}_{marker_id}"
                                transform.transform.translation.x = filtered_position[0]
                                transform.transform.translation.y = filtered_position[1]
                                transform.transform.translation.z = filtered_position[2]
                                transform.transform.rotation.x = filtered_quaternion[0]
                                transform.transform.rotation.y = filtered_quaternion[1]
                                transform.transform.rotation.z = filtered_quaternion[2]
                                transform.transform.rotation.w = filtered_quaternion[3]
                                
                                self.marker_info_pub.publish(marker_info)
                                self.tf_broadcaster.sendTransform(transform)
                                # Aggiungi questa linea per pubblicare anche il frame aggiustato
                                self.publish_adjusted_marker_frame(marker_id, filtered_position, filtered_quaternion)


                                # Pubblica un formato compatibile con face_tracker:
                                # Crea un bounding box simulato [x1, y1, x2, y2] usando il centro e l'area stimata
                                bbox_width = int(np.sqrt(marker_area))
                                bbox_height = bbox_width
                                x1 = center[0] - bbox_width//2
                                y1 = center[1] - bbox_height//2
                                x2 = center[0] + bbox_width//2
                                y2 = center[1] + bbox_height//2

                                # Formato compatibile con face_tracker
                                marker_detections[f"marker_{marker_id}"] = [int(x1), int(y1), int(x2), int(y2)]

                                
                                # Visualization
                                if self.display_detection:
                                    distance = np.sqrt(np.sum(filtered_position**2)) * 100  # in cm
                                    cv2.putText(display_frame,
                                                f"Dist: {distance:.1f}cm", 
                                                (center[0] - 40, center[1] - 20),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.6,
                                                (0, 255, 0),
                                                2)
                                    # Draw original and filtered axes
                                    display_frame = self.draw_axis(display_frame, rvec, tvec, (0, 0, 255))  # Original in red
                                    
                                    # Convert filtered quaternion and position back to rvec, tvec for visualization
                                    r_filtered = R.from_quat([filtered_quaternion[0], filtered_quaternion[1], 
                                                            filtered_quaternion[2], filtered_quaternion[3]])
                                    rot_matrix_filtered = r_filtered.as_matrix()
                                    rvec_filtered, _ = cv2.Rodrigues(rot_matrix_filtered)
                                    tvec_filtered = np.array([[filtered_position[0]], [filtered_position[1]], [filtered_position[2]]])
                                    
                                    display_frame = self.draw_axis(display_frame, rvec_filtered, tvec_filtered, (0, 255, 0))  # Filtered in green
                                    
                                    # Display Kalman state information
                                    cv2.putText(display_frame,
                                                f"KF active: stabilizing pose", 
                                                (10, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.7,
                                                (0, 255, 0),
                                                2)

                        except Exception as e:
                            rospy.logwarn(f"Pose estimation failed: {str(e)}")
                            continue
            
            # Pubblica le informazioni di rilevamento in formato JSON
            self.detections_pub.publish(String(json.dumps(marker_detections)))

            # Handle case when marker is not detected
            # if not target_detected:
            #     if self.marker_visible:
            #         # Marker was visible but now lost
            #         if self.marker_lost_time is None:
            #             self.marker_lost_time = rospy.Time.now()
                    
            #         # Check if we can still use Kalman prediction
            #         time_since_lost = rospy.Time.now() - self.marker_lost_time
            #         if time_since_lost < self.marker_redetection_timeout:
            #             # Get Kalman prediction without measurement update
            #             filtered_position, filtered_quaternion = self.pose_filter.get_current_estimate()
            #             filtered_position, filtered_quaternion = self.rotate_marker_axes(filtered_position, filtered_quaternion)

            #             # Create MarkerInfo message with predicted values
            #             marker_info = MarkerInfo()
            #             marker_info.id = self.target_marker_id
            #             marker_info.center_error = 0  # No real measurement
                        
            #             # Fill pose information with predicted values
            #             marker_info.pose.position.x = filtered_position[0]
            #             marker_info.pose.position.y = filtered_position[1]
            #             marker_info.pose.position.z = filtered_position[2]
            #             marker_info.pose.orientation.x = filtered_quaternion[0]
            #             marker_info.pose.orientation.y = filtered_quaternion[1]
            #             marker_info.pose.orientation.z = filtered_quaternion[2]
            #             marker_info.pose.orientation.w = filtered_quaternion[3]
                        
            #             # Update TF transformation
            #             transform = TransformStamped()
            #             transform.header.stamp = rospy.Time.now()
            #             transform.header.frame_id = self.base_frame
            #             transform.child_frame_id = f"{self.marker_frame}_{self.target_marker_id}"
            #             transform.transform.translation.x = filtered_position[0]
            #             transform.transform.translation.y = filtered_position[1]
            #             transform.transform.translation.z = filtered_position[2]
            #             transform.transform.rotation.x = filtered_quaternion[0]
            #             transform.transform.rotation.y = filtered_quaternion[1]
            #             transform.transform.rotation.z = filtered_quaternion[2]
            #             transform.transform.rotation.w = filtered_quaternion[3]
                        
            #             self.marker_info_pub.publish(marker_info)
            #             self.tf_broadcaster.sendTransform(transform)
                        
            #             # Aggiungi questa linea per pubblicare anche il frame aggiustato
            #             self.publish_adjusted_marker_frame(marker_id, filtered_position, filtered_quaternion)
            #             # Visualization
            #             if self.display_detection:
            #                 cv2.putText(display_frame,
            #                             f"Marker lost: using KF prediction", 
            #                             (10, 30),
            #                             cv2.FONT_HERSHEY_SIMPLEX,
            #                             0.7,
            #                             (0, 120, 255),  # Orange
            #                             2)
            #         else:
            #             # Marker lost for too long
            #             self.marker_visible = False
            #             if self.display_detection:
            #                 cv2.putText(display_frame,
            #                             f"Marker lost: tracking failed", 
            #                             (10, 30),
            #                             cv2.FONT_HERSHEY_SIMPLEX,
            #                             0.7,
            #                             (0, 0, 255),  # Red
            #                             2)
            
            # Publish detection result
            aruco_detected = Bool()
            aruco_detected.data = target_detected
            self.detection_pub.publish(aruco_detected)
            
            # Show detection if requested
            if self.display_detection:
                cv2.imshow('ArUco Detection with Kalman Filter', display_frame)
                cv2.waitKey(1)
            
            rate.sleep()

    def draw_axis(self, img, rvec, tvec, color_base=(0, 0, 255)):
        """Draw 3D coordinate axes on the marker with custom color."""
        axis_length = self.marker_size * 1.5  # Make axes longer for better visualization
        axis_points = np.float32([[0,0,0], 
                                [axis_length,0,0], 
                                [0,axis_length,0], 
                                [0,0,axis_length]])
        
        imgpts, jac = cv2.projectPoints(axis_points, rvec, tvec, 
                                      self.camera_matrix, self.dist_coeffs)
        
        corner = tuple(map(int, imgpts[0].ravel()))
        img = cv2.line(img, corner, tuple(map(int, imgpts[1].ravel())), (0, 0, 255), 3)  # X axis (red)
        img = cv2.line(img, corner, tuple(map(int, imgpts[2].ravel())), (0, 255, 0), 3)  # Y axis (green)
        img = cv2.line(img, corner, tuple(map(int, imgpts[3].ravel())), (255, 0, 0), 3)  # Z axis (blue)
        return img

    def rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion representation."""
        trace = R[0,0] + R[1,1] + R[2,2]
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2,1] - R[1,2]) / S
            qy = (R[0,2] - R[2,0]) / S
            qz = (R[1,0] - R[0,1]) / S
        else:
            if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
                S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
                qw = (R[2,1] - R[1,2]) / S
                qx = 0.25 * S
                qy = (R[0,1] + R[1,0]) / S
                qz = (R[0,2] + R[2,0]) / S
            elif R[1,1] > R[2,2]:
                S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
                qw = (R[0,2] - R[2,0]) / S
                qx = (R[0,1] + R[1,0]) / S
                qy = 0.25 * S
                qz = (R[1,2] + R[2,1]) / S
            else:
                S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
                qw = (R[1,0] - R[0,1]) / S
                qx = (R[0,2] + R[2,0]) / S
                qy = (R[1,2] + R[2,1]) / S
                qz = 0.25 * S
                
        return [qx, qy, qz, qw]
    

    def rotate_marker_axes(self, position, quaternion):
        """
        Ruota gli assi del marker in modo che:
        - La nuova X diventi l'attuale -Z
        - La nuova Z diventi l'attuale X
        
        Args:
            position: array [x, y, z]
            quaternion: array [qx, qy, qz, qw]
            
        Returns:
            position_rotated: array [x, y, z] ruotato
            quaternion_rotated: array [qx, qy, qz, qw] ruotato
        """
        # Converti il quaternione in una matrice di rotazione
        r = R.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
        rot_matrix = r.as_matrix()
        
        # Crea una matrice di rotazione per la trasformazione richiesta
        # Scambia l'asse X con l'asse Z e inverte l'asse Z
        axes_rotation = np.array([
            [-1, 0, 0],   # Nuova X = vecchia Z
            [0, -1, 0],   # Nuova Y = vecchia Y
            [0, 0, 1]   # Nuova Z = vecchia -X
        ])
        
        # Applica la rotazione alla matrice originale
        rot_matrix_rotated = np.dot(rot_matrix, axes_rotation)
        
        # Converti la nuova matrice di rotazione in un quaternione
        r_rotated = R.from_matrix(rot_matrix_rotated)
        quaternion_rotated = r_rotated.as_quat()  # [x, y, z, w]
        
        # Riorganizza per avere il formato [qx, qy, qz, qw]
        quaternion_rotated = np.array([
            quaternion_rotated[0],
            quaternion_rotated[1],
            quaternion_rotated[2],
            quaternion_rotated[3]
        ])
        
        # Ruota anche la posizione secondo la stessa logica se necessario
        # Nel caso del marker, in genere si vuole mantenere la stessa posizione dell'origine
        position_rotated = position.copy()
        
        return position_rotated, quaternion_rotated

    def publish_adjusted_marker_frame(self, marker_id, position, quaternion):
        """
        Pubblica un frame TF aggiustato con un offset rispetto al marker,
        mantenendo solo l'angolo di yaw e allineando pitch e roll al frame map.
        
        Args:
            marker_id: ID del marker
            position: posizione [x, y, z]
            quaternion: quaternione [qx, qy, qz, qw]
        """
        # Offset desiderato (ad es. 30cm lungo l'asse X del marker)
        offset = self.floating_offset  # in metri
        
        # Crea un oggetto di rotazione dal quaternione
        r = R.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
        
        # Calcola il vettore di offset nel sistema di riferimento globale
        # Questo aggiunge l'offset lungo l'asse X del marker (che punta fuori dal piano)
        offset_vector = r.apply([offset, 0, 0])
        
        # Applica l'offset alla posizione
        adjusted_position = position + offset_vector
        
        # Crea la trasformazione
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.base_frame
        transform.child_frame_id = f"floating_{self.marker_frame}_{marker_id}"
        
        # Imposta posizione e orientamento
        transform.transform.translation.x = adjusted_position[0]
        transform.transform.translation.y = adjusted_position[1]
        transform.transform.translation.z = adjusted_position[2]
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]
        
        # Pubblica la trasformazione
        self.tf_broadcaster.sendTransform(transform)

        # Publish notification that the floating marker frame is ready
        if self.should_process:
            floating_ready = Bool()
            floating_ready.data = True
            self.floating_ready_pub.publish(floating_ready)
            self.should_process = False
        
    def publish_map_aligned_marker_frame(self, marker_id, position):
        """
        Publishes a frame at the marker position but with the same orientation as the map frame.
        """
        try:
            # Try to get transform from camera to map
            trans = self.tf_buffer.lookup_transform(
                self.base_frame, self.map_frame, rospy.Time(0), rospy.Duration(0.1))
            
            # Extract map orientation relative to camera
            map_orientation = trans.transform.rotation
            
            # Create transform
            transform = TransformStamped()
            transform.header.stamp = rospy.Time.now()
            transform.header.frame_id = self.base_frame
            transform.child_frame_id = f"map_aligned_marker_{marker_id}"
            
            # Set position to marker position
            transform.transform.translation.x = position[0]
            transform.transform.translation.y = position[1]
            transform.transform.translation.z = position[2]
            
            # Set orientation to map orientation
            transform.transform.rotation = map_orientation
            
            
            # Publish the transformation
            self.tf_broadcaster.sendTransform(transform)

            return np.array([map_orientation.x, map_orientation.y, map_orientation.z, map_orientation.w])

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, 
                tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Cannot align with map frame: {e}")
            # Fall back to identity orientation if map frame is not available
            return np.array([0.0, 0.0, 0.0, 1.0])


    def cleanup(self):
        """Release camera resources and close windows."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    """Main entry point."""
    try:
        detector = ArucoDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'detector' in locals():
            detector.cleanup()