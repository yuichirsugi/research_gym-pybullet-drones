import pybullet as p
import pybullet_data
import time
import numpy as np

# PyBulletの初期化
physicsClient = p.connect(p.GUI)  # グラフィカルなGUIを使用
p.setGravity(0, 0, -9.8)  # 重力の設定

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeID = p.loadURDF("plane.urdf")

# # ドローンのモデルを読み込む（適切なURDFファイルを指定する必要があります）
# drone_urdf_path = "assets/c.urdf"
# drone_id = p.loadURDF(drone_urdf_path)  

# # カメラの視点を変更
# camera_distance = 0.25  # カメラとの距離
# camera_yaw = 60     # カメラの水平方向角度（yaw）
# camera_pitch = -20  # カメラの垂直方向角度（pitch）
# camera_target_position = [0, 0.2, 0.5]  # カメラの注視点

# # カメラの視点を設定
# p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

# 壁の初期位置とサイズ
wall_start_pos = [-0.5, 1, 0.5]
wall_size = [0.05, 1, 0.5]

# 壁を追加
wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=wall_size)
wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=wall_size)
wall_body = p.createMultiBody(baseVisualShapeIndex=wall_visual_shape, baseCollisionShapeIndex=wall_collision_shape, basePosition=wall_start_pos)

class DroneBltEnv():

    def __init__(self, sim_time_step):
        self._sim_time_step = sim_time_step
        self.drone_id = p.loadURDF("assets/c.urdf")  # DroneBltEnvクラスにdrone_idを初期化する


    
    def control_loop(self, target_pos, target_vel):
        # 現在の位置と速度を取得
        pos = np.array([0.0, 0.0, 2.0])  # 初期位置
        vel = np.array([0.0, 0.0, 0.0])  # 初期速度

        # # 位置制御
        # pos_error = target_pos - pos
        # accs = pos_error / self._sim_time_step

        # 速度制御
        vel_error = target_vel - vel
        accs = vel_error / self._sim_time_step

        # モーターに必要な力を計算
        m = 0.25  # 質量
        g = 9.8
        forces_world_frame = m * accs

        # モーターの力から角度を計算
        rpy = np.array([np.arctan(forces_world_frame[2] / forces_world_frame[0]),
                        np.arctan(forces_world_frame[2] / forces_world_frame[1]),
                        0])

        # 現在の位置と姿勢を取得
        current_pos, current_orn = p.getBasePositionAndOrientation(self.drone_id)

        # 新しい姿勢を計算
        new_orn = p.getQuaternionFromEuler(rpy)

        # 姿勢のみを適用
        p.resetBasePositionAndOrientation(self.drone_id, current_pos, new_orn)

# DroneBltEnvクラスのインスタンスを作成
env = DroneBltEnv(sim_time_step=1.0 / 240.0)  # シミュレーションのタイムステップを指定

# 目標位置を順に設定し、ドローンを動かす
move_points = [np.array([1.0, 2.0, 2.0]), np.array([-1.0, -1.0, 1.5]), np.array([0.0, 0.0, 0.0])]

for target_pos in move_points:
    target_vel = np.array([0.1, 0.1, 0.1])  # 速度は0.1 m/sとして指定
    env.control_loop(target_pos, target_vel)
    time.sleep(1.0)  # 1秒待つ（シミュレーションのタイムステップに合わせて調整）

# シミュレーションを終了
p.disconnect()