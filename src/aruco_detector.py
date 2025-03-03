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
class KalmanPoseFilter:
    def __init__(self, dt=1.0/15.0):  # 15Hz default rate
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
        
        # Measurement noise covariance (R) - AUMENTATO per ridurre l'influenza delle misurazioni rumorose
        self.kf.R = np.eye(7) * 0.03  # Aumentato da 0.01 a 0.03
        self.kf.R[3:, 3:] *= 0.005  # Aumentato da 0.001 a 0.005 per quaternioni
        
        # Process noise covariance (Q) - RIDOTTO per movimenti più fluidi
        self.kf.Q = np.eye(10) * 0.005  # Ridotto da 0.01 a 0.005
        self.kf.Q[3:6, 3:6] *= 0.05  # Ridotto da 0.1 a 0.05 per velocità
        self.kf.Q[6:, 6:] *= 0.0005  # Ridotto da 0.001 a 0.0005 per quaternioni
        
        # Initial state covariance (P)
        self.kf.P = np.eye(10) * 1.0
        self.kf.P[6:, 6:] *= 0.1  # Lower uncertainty for quaternion components
        
        # Initialize state (x)
        self.kf.x = np.zeros(10)
        self.kf.x[9] = 1.0  # Initial quaternion w=1 (identity rotation)
        
        # Buffer delle misurazioni recenti per l'outlier rejection
        self.position_buffer = []
        self.quaternion_buffer = []
        self.buffer_size = 5  # Mantieni le ultime 5 misurazioni
        
        self.initialized = False
        self.last_update_time = rospy.Time.now()
        
    def normalize_quaternion(self):
        """Normalize the quaternion in the state vector"""
        q = self.kf.x[6:10]
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            self.kf.x[6:10] = q / q_norm
    
    def is_outlier(self, position, quaternion):
        """Determina se una misurazione è un outlier basato sulla storia recente"""
        if len(self.position_buffer) < 3:
            return False
            
        # Calcola la media delle posizioni recenti
        avg_pos = np.mean(self.position_buffer, axis=0)
        # Distanza euclidea dalla posizione media
        pos_distance = np.linalg.norm(position - avg_pos)
        
        # Soglia dinamica basata sulla varianza delle posizioni precedenti
        pos_std = np.std([np.linalg.norm(p - avg_pos) for p in self.position_buffer])
        pos_threshold = max(0.05, 3.0 * pos_std)  # almeno 5cm o 3 deviazioni standard
        
        # Per la rotazione, usiamo il prodotto scalare del quaternione
        # Un prodotto scalare basso indica una grande rotazione
        quat_similarity = abs(np.dot(quaternion, self.kf.x[6:10]))
        quat_threshold = 0.95  # cos(18°) ≈ 0.95, quindi questa è una rotazione di circa 18 gradi
        
        # È un outlier se la posizione è troppo lontana o la rotazione è troppo grande
        return pos_distance > pos_threshold or quat_similarity < quat_threshold
    
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
        alpha = 0.8  # Fattore di smoothing (0.0-1.0), più alto = meno smoothing
        
        # Applica smoothing solo se abbiamo abbastanza campioni nel buffer
        if len(self.position_buffer) > 1:
            # Calcola la media pesata tra lo stato corrente e la media delle misurazioni recenti
            weighted_pos = alpha * self.kf.x[0:3] + (1-alpha) * np.mean(self.position_buffer, axis=0)
            
            # Applica il risultato filtrato allo stato
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
        self.display_detection = rospy.get_param('~display_detection', True)
        self.camera_id = rospy.get_param('~camera_id', 0)
        
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
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Original camera matrix for ZED Mini (right camera, 720p)
        self.camera_matrix = np.array([
            [343.14258235, 0.0, 336.03231532],    # fx, 0, cx
            [0.0, 342.54874283, 189.4767514],    # 0, fy, cy
            [0.0, 0.0, 1.0]             # 0, 0, 1
        ])
        
        # ZED Mini distortion coefficients
        self.dist_coeffs = np.array([[-1.88526242e-01, 7.27145339e-02, -9.95206423e-05, 1.09062011e-03, -2.70775321e-02]])
        
        # Set marker size in meters
        self.marker_size = rospy.get_param('~marker_size', 0.092)  # 9cm default
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # Frame names
        self.base_frame = "camera_left"
        self.marker_frame = "aruco_marker"
        
        # Variables to track marker detection
        self.marker_visible = False
        self.marker_lost_time = None
        self.marker_redetection_timeout = rospy.Duration(1.0)  # 1 second timeout
        
        rospy.loginfo(f"ArUco detection node started with Kalman filtering. Dictionary: {self.aruco_dictionary_name}")

    def run(self):
        """Main detection loop running at 15Hz."""
        rate = rospy.Rate(15)  # 15Hz
        
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
            
            # Handle case when marker is not detected
            if not target_detected:
                if self.marker_visible:
                    # Marker was visible but now lost
                    if self.marker_lost_time is None:
                        self.marker_lost_time = rospy.Time.now()
                    
                    # Check if we can still use Kalman prediction
                    time_since_lost = rospy.Time.now() - self.marker_lost_time
                    if time_since_lost < self.marker_redetection_timeout:
                        # Get Kalman prediction without measurement update
                        filtered_position, filtered_quaternion = self.pose_filter.get_current_estimate()
                        
                        # Create MarkerInfo message with predicted values
                        marker_info = MarkerInfo()
                        marker_info.id = self.target_marker_id
                        marker_info.center_error = 0  # No real measurement
                        
                        # Fill pose information with predicted values
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
                        transform.child_frame_id = f"{self.marker_frame}_{self.target_marker_id}"
                        transform.transform.translation.x = filtered_position[0]
                        transform.transform.translation.y = filtered_position[1]
                        transform.transform.translation.z = filtered_position[2]
                        transform.transform.rotation.x = filtered_quaternion[0]
                        transform.transform.rotation.y = filtered_quaternion[1]
                        transform.transform.rotation.z = filtered_quaternion[2]
                        transform.transform.rotation.w = filtered_quaternion[3]
                        
                        self.marker_info_pub.publish(marker_info)
                        self.tf_broadcaster.sendTransform(transform)
                        
                        # Visualization
                        if self.display_detection:
                            cv2.putText(display_frame,
                                        f"Marker lost: using KF prediction", 
                                        (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 120, 255),  # Orange
                                        2)
                    else:
                        # Marker lost for too long
                        self.marker_visible = False
                        if self.display_detection:
                            cv2.putText(display_frame,
                                        f"Marker lost: tracking failed", 
                                        (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 0, 255),  # Red
                                        2)
            
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