import pybullet as p
import time
import pybullet_data
import numpy as np
from typing import Tuple

# PyBulletの初期化
client = p.connect(p.GUI)
# p.resetSimulation() # 物体の位置や姿勢が初期状態にリセット

# PyBulletの設定
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

planeID = p.loadURDF("plane.urdf")

# 壁の初期位置とサイズ
wall_start_pos = [-0.5, 1, 0.5]
wall_size = [0.05, 1, 0.5]

# 壁を追加
wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=wall_size)
wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=wall_size)
wall_body = p.createMultiBody(baseVisualShapeIndex=wall_visual_shape, baseCollisionShapeIndex=wall_collision_shape, basePosition=wall_start_pos)

# ハケの長さ
brush_length = 0.1

# 壁の中心位置
wall_center_pos = np.array(wall_start_pos) + 0.5 * np.array(wall_size)

# ドローンが壁をなぞるための目標位置
target_position = [
    np.array([0, 0, wall_center_pos[2] + 0.25 * wall_size[2]]),  # 壁に近づく
    np.array([0, wall_center_pos[1], wall_center_pos[2] + 0.25 * wall_size[2]]),  # 壁をなぞる経路を指定
    np.array([wall_start_pos[0] + brush_length, wall_center_pos[1], wall_center_pos[2] + 0.25 * wall_size[2]]),
    np.array([wall_start_pos[0] + brush_length, wall_center_pos[1], wall_center_pos[2] + 0.25 * wall_size[2] - 0.5 * wall_size[2]]),
    np.array([wall_start_pos[0] + 2*brush_length, wall_center_pos[1], wall_center_pos[2] + 0.25 * wall_size[2] - 0.5 * wall_size[2]]),
    np.array([0, 0, 0]),
]

class PhysicsParameters:
    def __init__(self, kf, km, l, gf, J, J_inv, m):
        # ドローンの物理パラメータを初期化
        self.kf = 3.16e-10  # 推進力定数 8.3e-7
        self.km = 7.94e-12  # トルク定数 3.0e-8
        self.l = 0.1235  # アームの長さ
        self.gf = 2.44  # 重力 9.8 * 0.249（gf などの物理パラメータは URDF から取得できない場合に手動で設定）

        # ドローンの慣性モーメントのテンソル
        self.J = np.zeros((3, 3))

        # 慣性モーメントテンソルの逆行列
        self.J_inv = np.zeros_like(self.J)

        # ドローンの質量
        self.m = 0.249  

        # ドローンの寸法
        self.drone_dimensions = {
            'length': 0.251,  # ドローンの長さ
            'width': 0.362,   # ドローンの幅
            'height': 0.07    # ドローンの高さ
        }

        # 慣性モーメントテンソルを計算
        self.calculate_inertia_tensor()

    def calculate_inertia_tensor(self):
        # ドローンの寸法から慣性モーメントテンソルを計算
        length, width, height = self.drone_dimensions['length'], self.drone_dimensions['width'], self.drone_dimensions['height']

        # 質量分布に基づいて慣性モーメントテンソルを計算（単純なボックスの場合）
        self.J[0, 0] = (1 / 12) * self.m * (width**2 + height**2) # 0.002820837999999999
        self.J[1, 1] = (1 / 12) * self.m * (length**2 + height**2) # 0.0014089457499999998
        self.J[2, 2] = (1 / 12) * self.m * (length**2 + width**2) # 0.004026433749999999

        # 慣性モーメントテンソルの逆行列を計算
        self.J_inv = np.linalg.inv(self.J)

        # print("J_xx =", self.J[0, 0]) 
        # print("J_yy =", self.J[1, 1])
        # print("J_zz =", self.J[2, 2])


class DroneBltEnv:
    def __init__(self):
        # # PyBulletの初期化
        # self.client = p.connect(p.GUI)
        # # p.resetSimulation() # 物体の位置や姿勢が初期状態にリセット

        # # PyBulletの設定
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.setGravity(0, 0, -9.8)

        # ドローンのモデル読み込み
        self.drone_id = p.loadURDF("assets/a.urdf")

        # 初期状態の設定（例: 位置、姿勢）
        init_pos = [0, 0, 0]  #  初期位置 [x, y, z]
        init_orn = p.getQuaternionFromEuler([0, 0, 0])  #  初期姿勢 [roll, pitch, yaw]
        p.resetBasePositionAndOrientation(self.drone_id, init_pos, init_orn)

        # 物理パラメータの設定
        # self._dp = dp
        self._dp = PhysicsParameters(kf=3.16e-10, km=7.94e-12, l=0.1235, gf=9.8, J=np.zeros((3, 3)), J_inv=np.zeros((3, 3)), m=0.249)
        self._sim_time_step = 0.01

        # その他の初期化処理（例: 初期速度、物理パラメータの設定）
        self.pos = [0, 0, 0]
        self.quat = [0, 0, 0, 1]
        self.rpy = [0, 0, 0]
        self.vel = [0, 0, 0]
        self.rpy_rates = [0, 0, 0]
        # self.physics_params = {
        #     'kf': 1.0,  # 例: スラストに関する物理パラメータ
        #     'km': 0.5,  # 例: トルクに関する物理パラメータ
        #     # その他の物理パラメータ
        # }

    
    # DroneBltEnvクラスの関数
    def apply_rotor_physics(self, rpm: np.ndarray):
        #引数rpm: np.ndarrayは、角速度（rotations per minute）を表すNumPyの配列（numpy.ndarray）
        #rpmはrotations per minute（1分あたりの回転数）
        """
        慣性モーメントなどを考慮した力学を陽解法を用いて適用
        Parameters
        ----------
        rpm : A list with 4 elements.  Specify the 'rpm' for each of the four rotors.
	４つのロータの回転数を引数として与えて指定する
        """

    # current state
	# ドローンの現在位置、姿勢、速度などを取得
        # pos = self._ki.pos #クラス内の _ki というプロパティ（または変数）から pos という属性（またはメソッド）を取得して、変数 pos に代入
        # quat = self._ki.quat #四元数（quaternion）
        # rpy = self._ki.rpy # ロール（Roll）、ピッチ（Pitch）、ヨー（Yaw）のオイラー角
        # vel = self._ki.vel
        # rpy_rates = self._ki.rpy_rates # 角速度（angular rates）
        # rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        # current state
        pos = self.pos  # クラス内の pos というプロパティ（または変数）から pos という属性（またはメソッド）を取得して、変数 pos に代入
        quat = self.quat  # 四元数（quaternion）
        rpy = self.rpy  # ロール（Roll）、ピッチ（Pitch）、ヨー（Yaw）のオイラー角
        vel = self.vel
        rpy_rates = self.rpy_rates  # 角速度（angular rates）
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        # reshapeは、NumPyライブラリで提供される関数で、多次元配列の形状（shape）を変更するために使用


    # compute thrust and torques
	# 指定したローター回転数からスラストやトルクを計算
        thrust, x_torque, y_torque, z_torque = self.rpm2forces(rpm)
        thrust = np.array([0, 0, thrust])

        thrust_world_frame = np.dot(rotation, thrust) # np.dot(rotation, thrust): 行列の積を計算
        # ドローン座標系でのスラストがワールド座標系に変換されると、スラストの方向がドローンの姿勢に従って変化
        forces_world_frame = thrust_world_frame - np.array([0, 0, self._dp.gf])  # ワールド座標系でのスラストに対して、重力を補正

        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self._dp.J, rpy_rates)) # 角速度による慣性モーメントによるトルクを計算し、全体のトルクから差し引く
        rpy_rates_deriv = np.dot(self._dp.J_inv, torques) # 姿勢（ロール、ピッチ、ヨー）の角速度の時間変化を表すベクトル
        no_pybullet_dyn_accs = forces_world_frame / self._dp.m # PyBulletのシミュレーションを考慮せずに計算されたドローンの加速度

    # update state
	# 指定したステップ時間後の速度や姿勢を計算, self._sim_time_step:シミュレーションの時間ステップ
        self.vel = self.vel + self._sim_time_step * no_pybullet_dyn_accs
        self.rpy_rates = self.rpy_rates + self._sim_time_step * rpy_rates_deriv # 角速度（angular rates）
        self.pos = self.pos + self._sim_time_step * self.vel 
        self.rpy = self.rpy + self._sim_time_step * self.rpy_rates # ロール（Roll）、ピッチ（Pitch）、ヨー（Yaw）のオイラー角
	
	# Set PyBullet state
	# アニメーションとして表示させるために、計算結果をPyBulletに渡す
    # p.resetBasePositionAndOrientation 関数は、PyBulletのシミュレーション内で物体の位置と姿勢をリセットするための関数
        p.resetBasePositionAndOrientation(
            bodyUniqueId=self.drone_id,
            posObj=pos,
            ornObj=p.getQuaternionFromEuler(rpy), # 物体の姿勢（orientation）
            physicsClientId=self.client,
        )

        # Note: the base's velocity only stored and not used
        # physicsClientId は、PyBulletの物理エンジンとの接続を確立するためのクライアントIDです。
        # PyBulletは複数の物理エンジンを同時にサポートできるため、各物理エンジンとの接続を区別するために使用
        p.resetBaseVelocity(
            objectUniqueId=self.drone_id,
            linearVelocity=vel,
            angularVelocity=rpy_rates,  # ang_vel not computed by DYN [-1, -1, -1]
            physicsClientId=self.client,
        )

    def rpm2forces(self, rpm: np.ndarray) -> Tuple:
        # -> Tuple は、関数の返り値の型をアノテートしている
        # rpm2forces 関数はタプル型 (Tuple) を返すことが期待 
        # > Tuple は、この関数が4つの値（thrust, x_torque, y_torque, z_torque）からなるタプル型を返すことを示している（returnより）
        """
        Compute thrust and x, y, z axis torque at specified rotor speed.
	スラストやトルクを指定したローターの回転数から算出する
        """
        forces = np.array(rpm) ** 2 * self._dp.kf
        thrust = np.sum(forces)
        z_torques = np.array(rpm) ** 2 * self._dp.km
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        x_torque = (forces[1] - forces[3]) * self._dp.l
        y_torque = (-forces[0] + forces[2]) * self._dp.l
        return thrust, x_torque, y_torque, z_torque


# シミュレーションの実行
drone_env = DroneBltEnv()

# for _ in range(1000):
#     # 制御コードを追加
#     rotor_rpm = np.array([14000, 14000, 14000, 14000])# 4つのドローンのローターに適用される回転数
#     drone_env.apply_rotor_physics(rotor_rpm)
#     # drone_env というドローンの環境（おそらくクラスやオブジェクト）に対して、apply_rotor_physics 関数を呼び出している

#     # シミュレーションのステップを進める
#     p.stepSimulation()
#     time.sleep(1.0 / 240)

# # 目標位置
# target_position = np.array([0, 0, 5.0])  # 例: 上に移動する

# シミュレーションの実行
for _ in range(1000):
    # 簡単な制御アルゴリズム: 目標位置に向かってドローンを制御
    current_position = np.array(drone_env.pos[:3])  # 最初の3つの要素を取得
    print("current_position:", current_position)
    # 各目標位置に対する誤差を計算
    errors = [target - current_position for target in target_position]
    
    # 全体の誤差を計算
    total_error = np.concatenate(errors, axis=0)
    # 修正: total_errorからZ軸方向の誤差のみを取得
    z_axis_error = total_error[2::3]
    rotor_rpm = np.array([14000, 14000, 14000, 14000]) + 0.1 * z_axis_error # Z軸方向の誤差に比例して調整
    # 制御コードを追加
    drone_env.apply_rotor_physics(rotor_rpm)

    # シミュレーションのステップを進める
    p.stepSimulation()
    time.sleep(1.0 / 240)

# PyBulletの終了
p.disconnect()