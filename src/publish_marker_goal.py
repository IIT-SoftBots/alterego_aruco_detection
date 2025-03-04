import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class FixedMarkerPublisher:
    def __init__(self):
        rospy.init_node("fixed_marker_publisher")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        self.marker_frame = "floating_aruco_marker_23"
        self.fixed_frame = "odom"
        self.base_link_frame = "base_link"
        self.new_fixed_marker = "fixed_aruco_marker"
        
        self.pose_samples = []
        self.max_samples = 100
        self.stability_threshold = 0.01
        self.stable_transform = None
        self.height_adjusted_transform = None
        
        # Attendo che i frame necessari siano disponibili
        rospy.loginfo("Waiting for frames...")
        rospy.sleep(2.0)

    def extract_yaw(self, quaternion):
        """Estrae l'angolo di yaw da un quaternione."""
        # Converti il quaternione in una matrice di rotazione
        rot = R.from_quat([quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
        # Estrai gli angoli di Eulero (roll, pitch, yaw)
        euler = rot.as_euler('xyz')
        # Ritorna solo lo yaw
        return euler[2]
    
    def yaw_only_quaternion(self, yaw):
        """Crea un quaternione che rappresenta solo una rotazione di yaw."""
        # Crea un quaternione con roll e pitch a zero, solo yaw
        rot = R.from_euler('xyz', [0, 0, yaw])
        return rot.as_quat()

    def get_stable_transform(self):
        try:
            # Ottiene la trasformazione del marker
            trans = self.tf_buffer.lookup_transform(self.fixed_frame, self.marker_frame, rospy.Time(0), rospy.Duration(1.0))
            pos = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            quat = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
            
            # Estrai lo yaw dal quaternione del marker
            yaw = self.extract_yaw(quat)
            
            # Ottiene la posizione di base_link
            try:
                base_trans = self.tf_buffer.lookup_transform(self.fixed_frame, self.base_link_frame, rospy.Time(0), rospy.Duration(1.0))
                base_height = base_trans.transform.translation.z
                
                # Accumula la posizione con l'altezza di base_link e yaw originale
                adjusted_pos = np.array([pos[0], pos[1], base_height])
                # Crea un quaternione con solo lo yaw del marker (roll e pitch a zero)
                yaw_quat = self.yaw_only_quaternion(yaw)
                
                self.pose_samples.append((adjusted_pos, yaw_quat))
            except Exception as e:
                rospy.logwarn(f"Couldn't get base_link height: {e}. Using original marker height.")
                # Usa comunque solo lo yaw
                yaw_quat = self.yaw_only_quaternion(yaw)
                self.pose_samples.append((pos, yaw_quat))
            
            if len(self.pose_samples) > self.max_samples:
                self.pose_samples.pop(0)
            
            if len(self.pose_samples) == self.max_samples:
                pos_var = np.var([p[0] for p in self.pose_samples], axis=0)
                quat_var = np.var([p[1] for p in self.pose_samples], axis=0)
                
                if np.all(pos_var < self.stability_threshold) and np.all(quat_var < self.stability_threshold):
                    avg_pos = np.mean([p[0] for p in self.pose_samples], axis=0)
                    avg_quat = np.mean([p[1] for p in self.pose_samples], axis=0)
                    
                    # Normalizza il quaternione
                    quat_norm = np.linalg.norm(avg_quat)
                    if quat_norm > 0:
                        avg_quat = avg_quat / quat_norm
                    
                    # Crea la trasformazione fissa all'altezza di base_link e yaw del marker
                    stable_trans = TransformStamped()
                    stable_trans.header.frame_id = self.fixed_frame
                    stable_trans.child_frame_id = self.new_fixed_marker
                    stable_trans.transform.translation.x = avg_pos[0]
                    stable_trans.transform.translation.y = avg_pos[1]
                    stable_trans.transform.translation.z = avg_pos[2]  # Questa è già all'altezza di base_link
                    stable_trans.transform.rotation.x = avg_quat[0]
                    stable_trans.transform.rotation.y = avg_quat[1]
                    stable_trans.transform.rotation.z = avg_quat[2]
                    stable_trans.transform.rotation.w = avg_quat[3]
                    
                    # Calcola e logga l'angolo di yaw per informazione
                    yaw_deg = math.degrees(self.extract_yaw(avg_quat))
                    rospy.loginfo(f"Created fixed frame at base_link height: {avg_pos[2]:.3f}m, yaw: {yaw_deg:.1f}°")
                    return stable_trans
        except Exception as e:
            rospy.logwarn(f"Failed to get transform: {e}")
        return None

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.stable_transform is None:
                self.stable_transform = self.get_stable_transform()
                if self.stable_transform:
                    rospy.loginfo("Fixed transform established at base_link height with marker yaw. Publishing continuously.")
            
            if self.stable_transform:
                self.stable_transform.header.stamp = rospy.Time.now()
                self.tf_broadcaster.sendTransform(self.stable_transform)

            rate.sleep()

if __name__ == "__main__":
    try:
        FixedMarkerPublisher().run()
    except rospy.ROSInterruptException:
        pass