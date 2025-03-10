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
        
        # Parametri per il rilevamento del cambiamento di posizione
        self.update_threshold = rospy.get_param('~update_threshold', 0.05)  # 5cm di tolleranza
        self.angular_threshold = rospy.get_param('~angular_threshold', 0.1)  # ~6 gradi di tolleranza
        self.update_window = rospy.get_param('~update_window', 20)  # Campioni per rilevare un cambio
        self.update_stability_time = rospy.get_param('~update_stability_time', 2.0)  # Tempo di stabilità in secondi
        
        # Buffer recente per verificare cambi di posizione
        self.recent_positions = []
        self.recent_yaws = []
        self.recent_max_samples = 30
        
        # Flag e timer per il processo di aggiornamento
        self.updating = False
        self.update_start_time = None
        
        # Contatore per i log
        self.update_count = 0
        
        # Attendo che i frame necessari siano disponibili
        rospy.loginfo("Waiting for frames...")
        rospy.sleep(2.0)
        
        rospy.loginfo("Fixed marker publisher initialized with dynamic updating")
        rospy.loginfo(f"Update thresholds: position={self.update_threshold}m, angle={math.degrees(self.angular_threshold):.1f}°")

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

    def get_current_marker_pose(self):
        """Ottiene la posizione corrente del marker e la adatta con l'altezza di base_link e solo yaw."""
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
                
                return adjusted_pos, yaw_quat, yaw
                
            except Exception as e:
                rospy.logwarn(f"Couldn't get base_link height: {e}. Using original marker height.")
                # Usa comunque solo lo yaw
                yaw_quat = self.yaw_only_quaternion(yaw)
                return pos, yaw_quat, yaw
                
        except Exception as e:
            rospy.logwarn(f"Failed to get transform: {e}")
            return None, None, None

    def get_stable_transform(self):
        """Calcola una trasformazione stabile basata su un certo numero di campioni."""
        self.pose_samples = []  # Azzera i campioni
        self.updating = True
        self.update_start_time = rospy.Time.now()
        
        while len(self.pose_samples) < self.max_samples and (rospy.Time.now() - self.update_start_time).to_sec() < 10.0:
            pos, quat, _ = self.get_current_marker_pose()
            if pos is not None:
                self.pose_samples.append((pos, quat))
            rospy.sleep(0.05)  # Campiona a circa 20Hz
            
        if len(self.pose_samples) < self.max_samples / 2:
            rospy.logwarn(f"Insufficient samples ({len(self.pose_samples)}/{self.max_samples}), aborting update")
            self.updating = False
            return None
            
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
            stable_trans.transform.translation.z = avg_pos[2]
            stable_trans.transform.rotation.x = avg_quat[0]
            stable_trans.transform.rotation.y = avg_quat[1]
            stable_trans.transform.rotation.z = avg_quat[2]
            stable_trans.transform.rotation.w = avg_quat[3]
            
            # Calcola e logga l'angolo di yaw per informazione
            yaw_deg = math.degrees(self.extract_yaw(avg_quat))
            self.update_count += 1
            rospy.loginfo(f"Update #{self.update_count}: Fixed frame at ({avg_pos[0]:.3f}, {avg_pos[1]:.3f}), height: {avg_pos[2]:.3f}m, yaw: {yaw_deg:.1f}°")
            
            self.updating = False
            return stable_trans
        else:
            rospy.logwarn("Position too unstable for updating")
            self.updating = False
            return None

    def check_position_changed(self, pos, yaw):
        """Verifica se la posizione attuale è cambiata significativamente rispetto alla memorizzata."""
        if self.stable_transform is None:
            return False
            
        # Aggiungi la posizione corrente al buffer recente
        self.recent_positions.append(pos)
        self.recent_yaws.append(yaw)
        
        # Mantieni solo gli ultimi campioni
        if len(self.recent_positions) > self.recent_max_samples:
            self.recent_positions.pop(0)
            self.recent_yaws.pop(0)
            
        # Se non abbiamo abbastanza campioni, aspetta
        if len(self.recent_positions) < self.update_window:
            return False
            
        # Calcola la posizione media recente
        recent_pos_mean = np.mean(self.recent_positions[-self.update_window:], axis=0)
        recent_yaw_mean = np.mean(self.recent_yaws[-self.update_window:])
        
        # Estrai la posizione attualmente memorizzata
        fixed_pos = np.array([
            self.stable_transform.transform.translation.x,
            self.stable_transform.transform.translation.y,
            self.stable_transform.transform.translation.z
        ])
        
        fixed_quat = np.array([
            self.stable_transform.transform.rotation.x,
            self.stable_transform.transform.rotation.y,
            self.stable_transform.transform.rotation.z,
            self.stable_transform.transform.rotation.w
        ])
        fixed_yaw = self.extract_yaw(fixed_quat)
        
        # Calcola la distanza e la differenza di angolo
        pos_difference = np.linalg.norm(recent_pos_mean[:2] - fixed_pos[:2])  # Solo x,y
        yaw_difference = abs(self.normalize_angle(recent_yaw_mean - fixed_yaw))
        
        # Verifica se sono oltre le soglie
        pos_changed = pos_difference > self.update_threshold
        yaw_changed = yaw_difference > self.angular_threshold
        
        if pos_changed or yaw_changed:
            rospy.loginfo(f"Position change detected: distance={pos_difference:.3f}m (threshold={self.update_threshold}m), " +
                          f"angle={math.degrees(yaw_difference):.1f}° (threshold={math.degrees(self.angular_threshold):.1f}°)")
            return True
            
        return False
    
    def normalize_angle(self, angle):
        """Normalizza un angolo nell'intervallo [-pi, pi]."""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def run(self):
        rate = rospy.Rate(20)
        last_log_time = rospy.Time.now()
        
        while not rospy.is_shutdown():
            # Prima di tutto, pubblica sempre il marker fisso se esiste
            if self.stable_transform:
                self.stable_transform.header.stamp = rospy.Time.now()
                self.tf_broadcaster.sendTransform(self.stable_transform)
            
            # Inizializzazione iniziale se necessaria
            if self.stable_transform is None:
                self.stable_transform = self.get_stable_transform()
                if self.stable_transform:
                    rospy.loginfo("Initial fixed transform established. Will check for significant position changes.")
                    continue  # Passa alla prossima iterazione per pubblicare subito il transform
            
            # Se non stiamo già aggiornando, controlla se dobbiamo aggiornare
            # if not self.updating and self.stable_transform is not None:
            #     pos, _, yaw = self.get_current_marker_pose()
                
            #     if pos is not None:
            #         if self.check_position_changed(pos, yaw):
            #             # Avvia un processo di aggiornamento in un thread separato
            #             # per non interrompere la pubblicazione del marker esistente
            #             rospy.loginfo(f"Position change detected, starting update process...")
            #             self.updating = True
                        
            #             # Usa un thread separato per l'aggiornamento
            #             import threading
            #             update_thread = threading.Thread(target=self.update_transform)
            #             update_thread.daemon = True  # Il thread termina quando il programma principale termina
            #             update_thread.start()
            
            # Log periodico (ogni 10 secondi)
            if (rospy.Time.now() - last_log_time).to_sec() > 1000.0:
                if self.stable_transform:
                    fixed_pos = np.array([
                        self.stable_transform.transform.translation.x,
                        self.stable_transform.transform.translation.y
                    ])
                    fixed_quat = np.array([
                        self.stable_transform.transform.rotation.x,
                        self.stable_transform.transform.rotation.y,
                        self.stable_transform.transform.rotation.z,
                        self.stable_transform.transform.rotation.w
                    ])
                    yaw_deg = math.degrees(self.extract_yaw(fixed_quat))
                    status = "updating" if self.updating else "monitoring"
                    rospy.loginfo(f"Current fixed marker: ({fixed_pos[0]:.3f}, {fixed_pos[1]:.3f}), yaw: {yaw_deg:.1f}° [{status}]")
                last_log_time = rospy.Time.now()

            rate.sleep()

    def update_transform(self):
        """Metodo per aggiornare la trasformazione in modo asincrono."""
        # Attendi un breve periodo per assicurarsi che la nuova posizione sia stabile
        rospy.loginfo(f"Waiting {self.update_stability_time}s for stability before updating...")
        rospy.sleep(self.update_stability_time)
        
        # Aggiorna la trasformazione stabile
        new_transform = self.get_stable_transform()
        if new_transform:
            # Aggiorna il transform ma mantiene la pubblicazione continua
            self.stable_transform = new_transform
            
            # Azzera i buffer per evitare falsi positivi subito dopo l'aggiornamento
            self.recent_positions = []
            self.recent_yaws = []
        else:
            rospy.logwarn("Update failed, keeping previous transform")
        
        # In ogni caso, segnala che l'aggiornamento è terminato
        self.updating = False

if __name__ == "__main__":
    try:
        FixedMarkerPublisher().run()
    except rospy.ROSInterruptException:
        pass