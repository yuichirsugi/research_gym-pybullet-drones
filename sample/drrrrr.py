import pybullet as p
import pybullet_data
import time
import numpy as np

# PyBulletの初期化
physicsClient = p.connect(p.GUI)  # グラフィカルなGUIを使用
p.setGravity(0, 0, -9.8)  # 重力の設定

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeID = p.loadURDF("plane.urdf")

# ドローンのモデルを読み込む（適切なURDFファイルを指定する必要があります）
drone_urdf_path = "assets/c.urdf"
drone_id = p.loadURDF(drone_urdf_path)  

# 壁の初期位置とサイズ
wall_start_pos = [-0.5, 1, 0.5]
wall_size = [0.05, 1, 0.5]

# 壁を追加
wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=wall_size)
wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=wall_size)
wall_body = p.createMultiBody(baseVisualShapeIndex=wall_visual_shape, baseCollisionShapeIndex=wall_collision_shape, basePosition=wall_start_pos)


# ドローンの回転数（RPM)設定
rpms = np.array([14300, 14300, 14300, 14300])
m = 0.25
g = 9.8

np.array(hover_rpm) = ([ m * g , m * g, m * g, m * g])

thrust = np.array([0, 0, thrust])

rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
thrust_world_frame = np.dot(rotation, thrust) # ワールド座標系における機体の推力ベクトル
forces_world_frame = thrust_world_frame - np.array([0, 0, m * g]) # このベクトルは、機体がどの方向にどれだけの力がかかっているかを示す
accs = forces_world_frame / m # 加速度


m = 0.25
g = 9.8

pos += [0, 0, 0]
vel += [0, 0, 0]
accs = vel / self._sim_time_step
rpy_rates_deriv = rpy_rates / self._sim_time_step # 角速度
forces_world_frame = m * accs
rpy = np.arctan( )
np.array([0, 0, thrust])
rpy = np.array([np.arctan( forces_world_frame[2] / forces_world_frame[0] ),
                np.arctan( forces_world_frame[2] / forces_world_frame[1] ),
                0 ])



vel = vel + self._sim_time_step * accs # 速度
        rpy_rates = rpy_rates + self._sim_time_step * rpy_rates_deriv # 角速度
        pos = pos + self._sim_time_step * vel # 位置
        rpy = rpy + self._sim_time_step * rpy_rates # 角度

class DroneBltEnv():

    def __init__(
            self,
            velocity,
            target_pos,

    ):

    def set_targets(self, target_velocity, target_position):
        self._target_velocity = target_velocity
        self._target_position = target_position

    def rpm2forces(self, rpm):
        # RPMから推力への変換
        forces = (np.array(rpm) ** 2) * self._dp.kf
        thrust = np.sum(forces)
        z_torques = np.array(rpm) ** 2 * self._dp.km
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self._dp.l / np.sqrt(2))
        y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self._dp.l / np.sqrt(2))
        return thrust, x_torque, y_torque, z_torque

    def apply_dynamics(self, rpm):
        # 提供されたコードに基づく力学のコードをここに記述
        assert len(rpm) == 4, f"The length of rpm_values must be 4. currently it is {len(rpm)}."
        
        # Current state
        m = 0.02
        ki = self._kis
        pos = ki.pos
        quat = ki.quat
        rpy = ki.rpy
        vel = ki.vel
        rpy_rates = self._rpy_rates
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        # Compute thrust and torques
        thrust, x_torque, y_torque, z_torque = self.rpm2forces(rpm)
        thrust = np.array([0, 0, thrust])

        thrust_world_frame = np.dot(rotation, thrust) # ワールド座標系における機体の推力ベクトル
        forces_world_frame = thrust_world_frame - np.array([0, 0, self._dp.gf]) # このベクトルは、機体がどの方向にどれだけの力がかかっているかを示す

        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self._dp.J, rpy_rates)) # ジャイロスコープの効果を考慮して、トルクベクトルを修正.外積
        rpy_rates_deriv = np.dot(self._dp.J_inv, torques) # 修正されたトルクから角加速度を計算しています。慣性テンソルの逆行列 角加速度

        accs = forces_world_frame / m # 加速度

        # Update state
        vel = vel + self._sim_time_step * accs # 速度
        rpy_rates = rpy_rates + self._sim_time_step * rpy_rates_deriv # 角速度
        pos = pos + self._sim_time_step * vel # 位置
        rpy = rpy + self._sim_time_step * rpy_rates # 角度

        # Set PyBullet state
        p.resetBasePositionAndOrientation(
            bodyUniqueId=self._drone_ids,
            posObj=pos,
            ornObj=p.getQuaternionFromEuler(rpy),
            physicsClientId=self._client,
        )

        p.resetBaseVelocity(
            objectUniqueId=self._drone_ids,
            linearVelocity=vel,
            angularVelocity=[-1, -1, -1],
            physicsClientId=self._client,
        )

        # Store the roll, pitch, yaw rates for the next step
        self._rpy_rates = rpy_rates






move_points = [np.array([1.0, 2.0, 2.0]), np.array([-1.0, -1.0, 1.5]), np.array([0.0, 0.0, 0.0])]


# 制御ループ
for _ in range(1000):  # 1000ステップ分のシミュレーションを実行
    # ここに制御コードを追加（例：ランダムな力を加える）
    force = [0, 0, 10]
    p.applyExternalForce(drone_id, -1, force, [0, 0, 0], p.LINK_FRAME)  # ドローンに力を加える

    # シミュレーションを1ステップ進める
    p.stepSimulation()

    # 一時停止して見やすくする（任意）
    time.sleep(1.0 / 240)  # 240FPSのシミュレーションを仮定

# PyBulletの終了
p.disconnect()
