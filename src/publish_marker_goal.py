import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
from std_msgs.msg import Bool

class OneTimeFixedMarkerPublisher:
    def __init__(self):
        rospy.init_node("fixed_marker_publisher")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        # Get the robot name from the environment variable
        self.robot_name = rospy.get_param('~robot_name', 'robot_adriano')
        
        self.marker_frame = "floating_aruco_marker_23"
        self.fixed_frame = "map"
        self.base_link_frame = "base_link"
        self.new_fixed_marker = "fixed_aruco_marker"
        
        self.pose_samples = []
        self.max_samples = 100
        self.stability_threshold = 0.01
        self.stable_transform = None

        
        # Flag to track if we should start processing
        self.should_process = False
        
        # Subscribe to the floating marker ready topic
        rospy.Subscriber(f'/{self.robot_name}/floating_marker_ready', Bool, self.floating_ready_callback)
        self.new_target_pub = rospy.Publisher(f'/{self.robot_name}/new_target', Bool, queue_size=1)

        # Attendo che i frame necessari siano disponibili
        rospy.loginfo("Waiting for frames...")
        rospy.sleep(2.0)

        # Verifica che il frame map sia disponibile
        try:
            self.tf_buffer.lookup_transform("map", "base_link", rospy.Time(0), rospy.Duration(2.0))
            rospy.loginfo("Map frame is available")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"Map frame not immediately available: {e}. Will retry during execution.")
        
        rospy.loginfo("One-time fixed marker publisher initialized with map frame")
        
    def floating_ready_callback(self, msg):
        """Callback for the floating marker ready topic"""
        if msg.data and not self.should_process:
            rospy.loginfo("Floating marker ready signal received. Starting processing...")
            self.should_process = True

    def extract_yaw(self, quaternion):
        """Estrae l'angolo di yaw da un quaternione."""
        rot = R.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
        euler = rot.as_euler('xyz')
        return euler[2]
    
    def yaw_only_quaternion(self, yaw):
        """Crea un quaternione che rappresenta solo una rotazione di yaw."""
        rot = R.from_euler('xyz', [0, 0, yaw])
        return rot.as_quat()

    def get_current_marker_pose(self):
        """Ottiene la posizione corrente del marker e la adatta con l'altezza di base_link e solo yaw."""
        try:
            trans = self.tf_buffer.lookup_transform(self.fixed_frame, self.marker_frame, rospy.Time(0), rospy.Duration(1.0))
            pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            quat = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
            
            yaw = self.extract_yaw(quat)
            
            try:
                base_trans = self.tf_buffer.lookup_transform(self.fixed_frame, self.base_link_frame, rospy.Time(0), rospy.Duration(1.0))
                base_height = base_trans.transform.translation.z
                
                adjusted_pos = np.array([pos[0], pos[1], base_height])
                yaw_quat = self.yaw_only_quaternion(yaw)
                
                return adjusted_pos, yaw_quat, yaw
                
            except Exception as e:
                rospy.logwarn(f"Couldn't get base_link height: {e}. Using original marker height.")
                yaw_quat = self.yaw_only_quaternion(yaw)
                return pos, yaw_quat, yaw
                
        except Exception as e:
            rospy.logwarn(f"Failed to get transform: {e}")
            return None, None, None

    def get_stable_transform(self):
        """Calcola una trasformazione stabile basata su un certo numero di campioni."""
        self.pose_samples = []
        update_start_time = rospy.Time.now()
        
        rospy.loginfo("Collecting samples to establish a stable fixed marker position...")
        
        while len(self.pose_samples) < self.max_samples and (rospy.Time.now() - update_start_time).to_sec() < 10.0:
            pos, quat, _ = self.get_current_marker_pose()
            if pos is not None:
                self.pose_samples.append((pos, quat))
            rospy.sleep(0.05)
            
        if len(self.pose_samples) < self.max_samples / 2:
            rospy.logerr(f"Insufficient samples ({len(self.pose_samples)}/{self.max_samples}), couldn't establish fixed marker")
            return None
            
        pos_var = np.var([p[0] for p in self.pose_samples], axis=0)
        quat_var = np.var([p[1] for p in self.pose_samples], axis=0)
        
        if np.all(pos_var < self.stability_threshold) :
            avg_pos = np.mean([p[0] for p in self.pose_samples], axis=0)
            avg_quat = np.mean([p[1] for p in self.pose_samples], axis=0)
            
            # Normalizza il quaternione
            quat_norm = np.linalg.norm(avg_quat)
            if quat_norm > 0:
                avg_quat = avg_quat / quat_norm
            
            stable_trans = TransformStamped()
            stable_trans.header.frame_id = self.fixed_frame
            stable_trans.child_frame_id = self.new_fixed_marker
            stable_trans.transform.translation.x = avg_pos[0]
            stable_trans.transform.translation.y = avg_pos[1] 
            stable_trans.transform.translation.z = avg_pos[2]
            stable_trans.transform.rotation.x = avg_quat[0]
            stable_trans.transform.rotation.y = avg_quat[1]
            stable_trans.transform.rotation.z = avg_quat[2]
            stable_trans.transform.rotation.w = avg_quat[3]
            
            yaw_deg = math.degrees(self.extract_yaw(avg_quat))
            rospy.loginfo(f"Fixed marker established at ({avg_pos[0]:.3f}, {avg_pos[1]:.3f}), height: {avg_pos[2]:.3f}m, yaw: {yaw_deg:.1f}°")
            
            return stable_trans
        else:
            rospy.logerr("Marker position too unstable, couldn't establish fixed marker. Please try again.")
            return None

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            # Check if we should start processing
            if self.should_process:
                # Establish the fixed transform once
                self.stable_transform = self.get_stable_transform()
                
                if self.stable_transform is None:
                    rospy.logerr("Failed to establish fixed marker transform. Will wait for next trigger.")
                    self.should_process = False
                else:
                    rospy.loginfo("Fixed marker transform established. Publishing continuously.")
                    # Reset flag to prevent reprocessing unless new trigger arrives
                    self.should_process = False
                    
                    # Continue publishing the stable transform
                    while not rospy.is_shutdown():
                        if self.should_process:
                            rospy.loginfo("New trigger received. Will re-establish fixed marker.")
                            break
                        self.stable_transform.header.stamp = rospy.Time.now()
                        self.tf_broadcaster.sendTransform(self.stable_transform)
                        rate.sleep()
                        
                        self.new_target_pub.publish(Bool(data=True))
            
            self.new_target_pub.publish(Bool(data=False))
            
            
            rate.sleep()

if __name__ == "__main__":
    try:
        OneTimeFixedMarkerPublisher().run()
    except rospy.ROSInterruptException:
        pass