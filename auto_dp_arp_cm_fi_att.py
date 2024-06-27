from typing import Optional, List, Tuple, Union
import time
import random

from logging import getLogger, NullHandler

import numpy as np
import pybullet as p
import pybullet_data

from blt_env.bullet_base import BulletEnv

from util.data_definition import DroneProperties, DroneType, DroneKinematicsInfo, PhysicsType, DroneForcePIDCoefficients, DroneControlTarget
from util.file_tools import DroneUrdfAnalyzer

from scipy.optimize import nnls

import math
from scipy.spatial.transform import Rotation

from control.ctrl_base import DroneEnvControl

logger = getLogger(__name__)
logger.addHandler(NullHandler())

# # Logger class to store drone status (optional).
# from util.data_logger import DroneDataLogger

def real_time_step_synchronization(sim_counts, start_time, time_step):

    if time_step > .04 or sim_counts % (int(1 / (24 * time_step))) == 0: #.04:0.04のこと
        elapsed = time.time() - start_time #経過時間を計算
        if elapsed < (sim_counts * time_step): 
            time.sleep(time_step * sim_counts - elapsed) # 必要なだけスリープ（一時停止）して、期待される経過時間に追いつく


def load_drone_properties(file_path: str, d_type: DroneType) -> DroneProperties:
    # load_drone_properties関数はDroneProperties型の値を返す
    file_analyzer = DroneUrdfAnalyzer() # ドローンのURDFファイルを解析するための解析器を作成
    return file_analyzer.parse(file_path, int(d_type)) 
    # 指定されたURDFファイルのパスとドローンの種類を用いてドローンのプロパティを解析して返す


class DroneBltEnv(BulletEnv):

    def __init__(
            self,
            urdf_path: str,
            d_type: DroneType = DroneType.QUAD_PLUS,
            phy_mode: PhysicsType = PhysicsType.PYB,
            sim_freq: int = 240,
            aggr_phy_steps: int = 1,
            is_gui: bool = True,
            is_real_time_sim: bool = False,
            init_xyzs: Optional[Union[List, np.ndarray]] = None,
            init_rpys: Optional[Union[List, np.ndarray]] = None,
    ):
        
        super().__init__(is_gui=is_gui)
        self._drone_type = d_type
        self._urdf_path = urdf_path
        self._physics_mode = phy_mode

        self._dp = load_drone_properties(self._urdf_path, self._drone_type)
        # self.printout_drone_properties()

        # PyBullet simulation settings. PyBulletシミュレーション設定
        self._aggr_phy_steps = aggr_phy_steps # 物理シミュレーションのステップ数
        self._g = self._dp.g # 重力の値
        self._sim_freq = sim_freq # PyBulletステップシミュレーションの頻度
        self._sim_time_step = 1. / self._sim_freq # シミュレーションのタイムステップ
        self._is_realtime_sim = is_real_time_sim  # add wait time in step(). # step()において待機時間を追加するかどうか

        # Initialization position of the drones. ドローンの初期位置
        if init_xyzs is None:
            self._init_xyzs = np.vstack([ # np.vstack 配列の結合　行が増える
                0, # X座標
                0, # Y座標
                (self._dp.collision_h / 2 - self._dp.collision_z_offset + 0.1), # np.ones 全ての要素が1の配列
            ]).transpose().reshape(3) # Z座標
        else: # もし初期位置が指定されている場合
            assert init_xyzs.ndim == 2, f"'init_xyzs' should has 2 dimension. current dims are {init_xyzs.ndim}."
            self._init_xyzs = np.array(init_xyzs) # 初期位置を指定された値にセット

        if init_rpys is None: # もし初期姿勢が指定されていない場合は、ゼロ行列で初期化
            self._init_rpys = np.zeros(3)
        else: # もし初期姿勢が指定されている場合
            assert init_rpys.ndim == 2, f"'init_rpys' should has 2 dimension. current dims are {init_rpys.ndim}."
            self._init_rpys = np.array(init_rpys)  # 初期姿勢を指定された値にセット

        # Simulation status. シミュレーションの状態
        self._sim_counts = 0 # シミュレーションの実行回数
        self._last_rpm_values = np.zeros(4) # 直前のrpm_values（ドローンの回転速度）の初期化

        self._kis = [DroneKinematicsInfo()]

        # もし物理モードがDYN（動力学モード）の場合は、角速度を保持する変数を初期化
        if self._physics_mode == PhysicsType.DYN: 
            self._rpy_rates = np.zeros(3)

        # PyBullet environment. PyBullet環境
        self._client = p.connect(p.GUI) if self._is_gui else p.connect(p.DIRECT)
        p.setGravity(0, 0, -self._g, physicsClientId=self._client)
        p.setRealTimeSimulation(0, physicsClientId=self._client)
        p.setTimeStep(self._sim_time_step, physicsClientId=self._client)

        # Load objects. オブジェクトの読み込み
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = p.loadURDF('plane.urdf')

        # カメラの初期設定
        p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

        # 壁の初期位置とサイズ
        wall_start_pos = [-0.5, 1, 0.5] #ハケの長さ 0.1
        # wall_size = [0.05, 1, 0.5]

        self.wall_id = p.loadURDF('./assets/wall_friction.urdf', basePosition=wall_start_pos)

        # Load drones. 各ドローンの読み込み
        self._drone_ids = p.loadURDF(
                self._urdf_path, # URDFファイルのパス
                self._init_xyzs, # ドローンの初期位置
                p.getQuaternionFromEuler(self._init_rpys), # ドローンの初期姿勢（オイラー角からクォータニオンに変換）
            ) 

        # Update the information before running the simulations.
        # シミュレーションを実行する前に情報を更新
        self.update_drones_kinematic_info()

        # Start measuring time. 時間の計測を開始
        self._start_time = time.time()

    def get_sim_time_step(self) -> float: # シミュレーションの時間ステップを取得
        return self._sim_time_step

    def get_sim_counts(self) -> int: # 現在のシミュレーションの反復回数を取得
        return self._sim_counts

    def get_drone_properties(self) -> DroneProperties: # ドローンのプロパティを取得 data_definition.py
        return self._dp

    def get_drones_kinematic_info(self) -> List[DroneKinematicsInfo]: # ドローンの運動情報のリストを取得
        return self._kis

    def get_aggr_phy_steps(self) -> int: # 物理ステップの集約数を取得
        return self._aggr_phy_steps

    def get_sim_freq(self) -> int: # シミュレーションの周波数を取得
        return self._sim_freq

    def get_num_drones(self) -> int: # ドローンの数を取得
        return self._num_drones 

    def get_last_rpm_values(self) -> np.ndarray: # 最後のRPM値を取得
        return self._last_rpm_values

    def get_drone_inf(self) : # 新しく定義
        return self._drone_ids, self.wall_id

    def get_gravity_and_mass(self) -> Tuple[float, float]:
        return self._dp.m, self._dp.g


    def update_drones_kinematic_info(self): # def init step
        pos, quat = p.getBasePositionAndOrientation( # ドローンの位置と姿勢を取得
            bodyUniqueId=self._drone_ids,
            physicsClientId=self._client,
        )
        rpy = p.getEulerFromQuaternion(quat) # クォータニオンからロールピッチヨー角を取得
        vel, ang_vel = p.getBaseVelocity( # ドローンの速度と角速度を取得
            bodyUniqueId=self._drone_ids,
            physicsClientId=self._client,
        )
        self._kis = DroneKinematicsInfo( # ドローンの運動情報を更新
            pos=np.array(pos),
            quat=np.array(quat),
            rpy=np.array(rpy),
            vel=np.array(vel),
            ang_vel=np.array(ang_vel),
        )

    def close(self) -> None:  # シミュレーションの終了時
        if p.isConnected() != 0:
            p.disconnect(physicsClientId=self._client) # PyBulletの接続を解除する。

    def reset(self) -> None: 
        if p.isConnected() != 0:
            p.resetSimulation(physicsClientId=self._client) # シミュレーションをリセットする。
            self.refresh_bullet_env() # PyBulletのパラメータやオブジェクトを初期化し、ドローンの運動情報を更新

    def step(self, rpm_values: np.ndarray) -> List[DroneKinematicsInfo]: # ドローンを指定された回転速度で1ステップ進める。
        """
        Parameters
        ----------
        rpm_values : Multiple arrays with 4 values as a pair of element. 4つの値からなる複数の配列のペア。
                    Specify the rotational speed of the four rotors of each drone. 各ドローンの4つのローターの回転速度を指定する。
        """
        rpm_values = self.check_values_for_rotors(rpm_values)

        for _ in range(self._aggr_phy_steps):
            '''
            Update and store the drones kinematic info the same action value of "rpm_values" 
            for the number of times specified by "self._aggr_phy_steps".
            "self._aggr_phy_steps"で指定された回数だけ、同じ"rpm_values"のアクション値で
            ドローンの運動情報を更新して保存する。
            '''
            if self._aggr_phy_steps > 1 and self._physics_mode in [
                PhysicsType.DYN,
                PhysicsType.PYB_GND,
                PhysicsType.PYB_DRAG,
                PhysicsType.PYB_DW,
                PhysicsType.PYB_GND_DRAG_DW
            ]:
                self.update_drones_kinematic_info()

            # step the simulation シミュレーションを進める
            self.physics(
                rpm_values[0, :],
                self._last_rpm_values[:],
            )

            # In the case of the explicit solution technique, 'p.stepSimulation()' is not used.
            # 明示的な解法テクニックの場合、'p.stepSimulation()'は使用されません。
            if self._physics_mode != PhysicsType.DYN:
                p.stepSimulation(physicsClientId=self._client)

            # Save the last applied action (for compute e.g. drag)
            # 最後に適用されたアクションを保存（ドラッグの計算などに使用)
            self._last_rpm_values = rpm_values

        # Update and store the drones kinematic information ドローンの運動情報を更新して保存
        self.update_drones_kinematic_info()

        # Advance the step counter ステップカウンターを進める
        self._sim_counts = self._sim_counts + (1 * self._aggr_phy_steps)

        # Synchronize the step interval with real time. リアルタイムでシミュレーションを同期させる
        if self._is_realtime_sim:
            real_time_step_synchronization(self._sim_counts, self._start_time, self._sim_time_step)

        return self._kis

    def check_values_for_rotors(self, rpm_values: np.ndarray) -> np.ndarray: #必要 メソッドstep
  
        cls_name = self.__class__.__name__
        assert isinstance(rpm_values, np.ndarray), f"Invalid rpm_values type is used on {cls_name}." # f"{cls_name}で無効なrpm_valuesの型が使用されています。
        assert rpm_values.ndim == 1 or rpm_values.ndim == 2, f"Invalid dimension of rpm_values is used on {cls_name}." # f"{cls_name}で無効なrpm_valuesの次元が使用されています。"
        if rpm_values.ndim == 1:
            assert len(rpm_values) == 4, f"Invalid number of elements were used for rpm_values on {cls_name}." # f"{cls_name}で無効なrpm_valuesの要素数が使用されています。"
            ''' e.g. 例
            while, a = [100, 200, 300, 400]
            then, np.tile(a, (3, 1)) -> [[100, 200, 300, 400], [100, 200, 300, 400], [100, 200, 300, 400]]
            '''
            rpm_values = np.tile(rpm_values, (1, 1))
        elif rpm_values.ndim == 2:
            assert rpm_values.shape[1] == 4, f"Invalid number of elements were used for rpm_values on {cls_name}." # f"{cls_name}で無効なrpm_valuesの要素数が使用されています。"
            rpm_values = np.reshape(rpm_values, (1, 4))
        return rpm_values

    def physics(
            self,
            rpm: np.ndarray,
            last_rpm: Optional[np.ndarray],
    ) -> None:

        def pyb(rpm, last_rpm=None):
            self.apply_rotor_physics(rpm)

        def dyn(rpm, last_rpm=None):
            self.apply_dynamics(rpm)

        def pyb_gnd(rpm, last_rpm=None):
            self.apply_rotor_physics(rpm)
            self.apply_ground_effect(rpm)

        def pyb_drag(rpm, last_rpm):
            self.apply_rotor_physics(rpm)
            self.apply_drag(last_rpm)  # apply last data

        def pyb_dw(rpm, last_rpm=None):
            self.apply_rotor_physics(rpm)
            self.apply_downwash()

        def pyb_gnd_drag_dw(rpm, last_rpm):
            self.apply_rotor_physics(rpm)
            self.apply_ground_effect(rpm)
            self.apply_drag(last_rpm)  # apply last data
            self.apply_downwash()

        def other(rpm, last_rpm):
            logger.error(f"In {self.__class__.__name__}, invalid physic mode key.") # 無効な物理モードキーです

        phy_key = self._physics_mode.value

        key_dict = {
            'pyb': pyb, # PyBullet物理演算
            'dyn': dyn, # 動力学モデル
            'pyb_gnd': pyb_gnd, # PyBullet物理演算 + 地面効果
            'pyb_drag': pyb_drag, # PyBullet物理演算 + 抗力
            'pyb_dw': pyb_dw, # PyBullet物理演算 + ダウンウォッシュ
            'pyb_gnd_drag_dw': pyb_gnd_drag_dw,  # PyBullet物理演算 + 地面効果 + 抗力 + ダウンウォッシュ
        }
        return key_dict.get(phy_key, other)(rpm, last_rpm)

    def apply_rotor_physics(self, rpm: np.ndarray):
        """
        Apply the individual thrusts and torques generated by the motion of the four rotors.
        4つのローターの動きによって発生する個々の推力とトルクを単純に適用

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        """
        assert len(rpm) == 4, f"The length of rpm_values must be 4. currently it is {len(rpm)}." # f"rpm_valuesの長さは4でなければなりません。現在の長さは {len(rpm)} です。"
        # 各ローターの推力とトルクを計算

        forces = (np.array(rpm) ** 2) * self._dp.kf
        torques = (np.array(rpm) ** 2) * self._dp.km
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])

        x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self._dp.l / np.sqrt(2))
        y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self._dp.l / np.sqrt(2))

        # 各ローターに推力を適用
        for i in range(4):
            p.applyExternalForce(
                objectUniqueId=self._drone_ids,
                linkIndex=i,  # link id of the rotors.　ローターのリンクID。
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self._client,
            )
        p.applyExternalTorque(
            objectUniqueId=self._drone_ids,
            linkIndex=4,  # link id of the center of mass.　重心のリンクID
            torqueObj=[x_torque, y_torque, z_torque],
            flags=p.LINK_FRAME,
            physicsClientId=self._client,
        )

    def printout_drone_properties(self) -> None:
        mes = f"""
        {self.__class__.__name__} loaded parameters from the .urdf :
        {self._urdf_path}
        m:ドローンの質量{self._dp.m} 
        l:ローターからベースまでのアームの長さ{self._dp.l}
        ixx:X軸周りの慣性モーメント {self._dp.ixx}
        iyy:Y軸周りの慣性モーメント {self._dp.iyy}
        izz:Z軸周りの慣性モーメント {self._dp.izz}
        kf:ローター推力定数 {self._dp.kf}
        km:ロータートルク定数 {self._dp.km}
        J:慣性モーメント行列 {self._dp.J}
        thrust2weight_ratio:推力と重量の比 {self._dp.thrust2weight_ratio}
        max_speed_kmh:最大速度（km/h） {self._dp.max_speed_kmh}
        gnd_eff_coeff:地面効果係数 {self._dp.gnd_eff_coeff}
        prop_radius:ローター半径 {self._dp.prop_radius}
        drag_coeff_xy:XY方向の抗力係数 {self._dp.drag_coeff_xy}
        drag_z_coeff:Z方向の抗力係数 {self._dp.drag_coeff_z}
        dw_coeff_1:ダウンワッシュ係数1 {self._dp.dw_coeff_1}
        dw_coeff_2:ダウンワッシュ係数2 {self._dp.dw_coeff_2}
        dw_coeff_3:ダウンワッシュ係数3 {self._dp.dw_coeff_3}
        gf:重力加速度 {self._dp.gf}
        hover_rpm:ホバリング時のローター回転数 {self._dp.hover_rpm}
        max_rpm:最大ローター回転数 {self._dp.max_rpm}
        max_thrust:最大推力 {self._dp.max_thrust}
        max_xy_torque:XY方向の最大トルク {self._dp.max_xy_torque}
        max_z_torque:Z方向の最大トルク {self._dp.max_z_torque}
        grand_eff_h_clip:地面効果の高さの制限 {self._dp.grand_eff_h_clip}
        grand_eff_h_clip: {self._dp.grand_eff_h_clip}
        A:ダウンワッシュモデル係数A {self._dp.A}
        B_coeff:ダウンワッシュモデル係数B {self._dp.B_coeff}
        Mixer:モーターミキシング行列 {self._dp.Mixer}
        """
        logger.info(mes)

def compute_rpm_by_nnls(
        thrust: float,
        x_torque: float,
        y_torque: float,
        z_torque: float,
        b_coeff: np.ndarray,
        a: np.ndarray,
        inv_a: np.ndarray = None,
):

    B = np.multiply(np.array([thrust, x_torque, y_torque, z_torque]), b_coeff) # b_coeff : 制御入力業行列の係数
    inv_a = np.linalg.inv(a) if inv_a is None else inv_a # linalg : 逆行列の計算
    sq_rpm = np.dot(inv_a, B)
    # sq_rpmは4つのモータのそれぞれの回転数を2乗したものが格納されているという前提
    # ただし、普通に行列式を解くとマイナスになる解が出る場合がある。回転数を2乗したものを解としたいのでこれは都合が悪い。
    # そのような場合に、解の全てを正にするという制約を与えて回帰する方法としてNNLSがある。
    sq_rpm_nnls, res = None, None
    # NNLS if any of the desired ang vel is negative
    if np.min(sq_rpm) < 0:
        sq_rpm_nnls, res = nnls(a, B, maxiter=3 * a.shape[1])
    # nnls は、与えられた行列やベクトルに対して非負制約を満たすように最小二乗法を適用する

    return sq_rpm, sq_rpm_nnls, res


class ForceControl(DroneEnvControl):

    def __init__(self, env: DroneBltEnv):
        super().__init__(env)
        self._is_gui = self._env.get_is_gui() # グラフィカルユーザーインターフェース(GUI)が有効かどうかのフラグ。
        self._time_step = self._env.get_sim_time_step() # シミュレーションで使用されるタイムステップ。
        self._dp = self._env.get_drone_properties() # 環境から取得したドローンのプロパティ。
        self._g = self._dp.g # 重力加速度。
        self._mass = self._dp.m # ドローンの質量。
        self._kf = self._dp.kf # モータの力の定数。
        self._km = self._dp.km # モータのトルクの定数。
        self._max_thrust = self._dp.max_thrust # ドローンが生成できる最大推力。
        self._max_xy_torque = self._dp.max_xy_torque # xy平面での最大トルク。
        self._max_z_torque = self._dp.max_z_torque # z軸まわりの最大トルク。
        self._A = self._dp.A # 制御計算で使用される係数行列A。
        self._inv_A = self._dp.inv_A # 係数行列Aの逆行列。
        self._B_coeff = self._dp.B_coeff # 制御計算で使用される係数。

    def value_check(
            self,
            thrust: float,
            x_torque: float,
            y_torque: float,
            z_torque: float,
            counter: int = 0,  # Simulation or control iteration, only used for printouts. シミュレーションまたは制御のイテレーション（出力のために使用）。
    ) -> None:
        if not self._is_gui:
            return

        name = self.__class__.__name__ # 制御入力が実現可能な範囲外の場合に警告をログに記録
        if thrust < 0 or thrust > self._max_thrust:
            mes = f"iter {counter} : in {name}, unfeasible thrust {thrust:.3f} \
            outside range [0, {self._max_thrust:.2f}]"
            logger.warning(mes)
        if np.abs(x_torque) > self._max_xy_torque:
            mes = f"iter {counter} : in {name}, unfeasible x_torque {x_torque:.2f} \
            outside range [{-self._max_xy_torque:.2f}, {self._max_xy_torque:.2f}]"
            logger.warning(mes)
        if np.abs(y_torque) > self._max_xy_torque:
            mes = f"iter {counter} : in {name}, unfeasible y_torque {y_torque:.2f} \
            outside range [{-self._max_xy_torque:.2f}, {self._max_xy_torque:.2f}]"
            logger.warning(mes)
        if np.abs(z_torque) > self._max_z_torque:
            mes = f"iter {counter} : in {name}, unfeasible z_torque {z_torque:.2f} \
            outside range [{-self._max_z_torque:.2f}, {self._max_z_torque:.2f}]"
            logger.warning(mes)

    def compute_control( # 望ましい推力およびトルクに基づいて制御入力を計算します。
        self,
        target_thrust: float,
        target_x_torque: float,
        target_y_torque: float,
        target_z_torque: float,
    ):
        counts = self._env.get_sim_counts()
        # 制御入力が実現可能な範囲内か確認
        self.value_check(target_thrust, target_x_torque, target_y_torque, target_z_torque, counts)

        sq_rpm, sq_nnls, res = compute_rpm_by_nnls( # 一番上のメソッド
            thrust=target_thrust,
            x_torque=target_x_torque, # 目標値をcompute_rpm_by_nnlsに渡す
            y_torque=target_y_torque,
            z_torque=target_z_torque,
            b_coeff=self._B_coeff,
            a=self._A,
            inv_a=self._inv_A,
        )
        # もしNNLS計算が失敗した場合、Noneを返す
        if sq_nnls is None:
            return np.sqrt(sq_rpm), None, None

        if self._is_gui: # 制御計算に関する情報をログに記録　不正確なローター速度が検出された場合　適切な解を得るプロセス
            name = self.__class__.__name__
            norm_1 = np.linalg.norm(sq_rpm)
            norm_2 = np.linalg.norm(sq_nnls)
            mes = f"""
            iter {counts} : in {name}, unfeasible squared rotor speeds, using NNLS.
            <Negative rotor speeds> # 負のローター速度
            sq. rotor speeds : [ {sq_rpm[0]}, {sq_rpm[1]}, {sq_rpm[2]}, {sq_rpm[3]} ] # 二乗ローター速度
            Normalized : [{sq_rpm[0] / norm_1}, {sq_rpm[1] / norm_1}, {sq_rpm[2] / norm_1}, {sq_rpm[3] / norm_1} ]
            <NNLS rotor speeds>
            sq. rotor speeds : [ {sq_nnls[0]}, {sq_rpm[1]}, {sq_rpm[2]}, {sq_rpm[3]} ] 
            Normalized : [ {sq_nnls[0] / norm_2}, {sq_rpm[1] / norm_2}, {sq_rpm[2] / norm_2}, {sq_rpm[3] / norm_2} ]  
            Residual : {res}
            """

        return np.sqrt(sq_nnls), None, None # ここでルートをつける n : 回転数


class DSLPIDControl(DroneEnvControl): # ドローンの位置および姿勢を制御するためのDSL-PIDコントローラクラス。

    def __init__(
            self,
            env: DroneBltEnv,
            pid_coeff: DroneForcePIDCoefficients,
    ):
        super().__init__(env)

        # PID constant parameters
        self._PID = pid_coeff

        self._pwm2rpm_scale = 0.2685
        self._pwm2rpm_const = 4070.3
        self._min_pwm = 20000
        self._max_pwm = 65535

        self._time_step = self._env.get_sim_time_step()
        self._dp = self._env.get_drone_properties()
        self._g = self._dp.g
        self._mass = self._dp.m
        self._kf = self._dp.kf
        self._km = self._dp.km
        self._Mixer = self._dp.Mixer
        self._gf = self._dp.gf

        # Initialized PID control variables
        self._last_rpy = np.zeros(3)
        self._last_pos_e = np.zeros(3)
        self._integral_pos_e = np.zeros(3)
        self._last_rpy_e = np.zeros(3)
        self._integral_rpy_e = np.zeros(3)

    def get_PID(self) -> DroneForcePIDCoefficients:
        return self._PID

    def set_PID(self, pid_coeff: DroneForcePIDCoefficients):
        self._PID = pid_coeff

    def reset(self):
        self._last_rpy = np.zeros(3)
        self._last_pos_e = np.zeros(3)
        self._integral_pos_e = np.zeros(3)
        self._last_rpy_e = np.zeros(3)
        self._integral_rpy_e = np.zeros(3)

    def compute_control_from_kinematics(
            self,
            control_timestep: float,
            kin_state: DroneKinematicsInfo,
            ctrl_target: DroneControlTarget,
    ) -> Tuple:
        """ Computes the PID control action (as RPMs) for a single drone. 単一のドローンに対するPID制御アクション（RPMとして）を計算します。
        Parameters
        ----------
        control_timestep: The time step at which control is computed. 制御が計算されるタイムステップ
        kin_state ドローンの運動学情報。
        ctrl_target 制御対象の情報。
        """
        return self.compute_control(
            control_timestep=control_timestep,
            current_position=kin_state.pos,
            current_quaternion=kin_state.quat,
            current_velocity=kin_state.vel,
            current_ang_velocity=kin_state.ang_vel,
            target_position=ctrl_target.pos,
            target_velocity=ctrl_target.vel,
            target_rpy=ctrl_target.rpy,
            target_rpy_rates=ctrl_target.rpy_rates,
        )

    def coating_attitude_control_from_kinematics(
            self,
            control_timestep: float,
            kin_state: DroneKinematicsInfo,
            ctrl_target: DroneControlTarget,
    ) -> Tuple:
        """ Computes the PID control action (as RPMs) for a single drone. 単一のドローンに対するPID制御アクション（RPMとして）を計算します。
        Parameters
        ----------
        control_timestep: The time step at which control is computed. 制御が計算されるタイムステップ
        kin_state ドローンの運動学情報。
        ctrl_target 制御対象の情報。
        """
        return self.coating_attitude_control(
            control_timestep=control_timestep,
            current_position=kin_state.pos,
            current_quaternion=kin_state.quat,
            current_velocity=kin_state.vel,
            current_ang_velocity=kin_state.ang_vel,
            target_position=ctrl_target.pos,
            target_velocity=ctrl_target.vel,
            target_rpy=ctrl_target.rpy,
            target_rpy_rates=ctrl_target.rpy_rates,
        )
    
    def coating_motion_control_from_kinematics(
            self,
            control_timestep: float,
            kin_state: DroneKinematicsInfo,
            ctrl_target: DroneControlTarget,
    ) -> Tuple:
        """ Computes the PID control action (as RPMs) for a single drone. 単一のドローンに対するPID制御アクション（RPMとして）を計算します。
        Parameters
        ----------
        control_timestep: The time step at which control is computed. 制御が計算されるタイムステップ
        kin_state ドローンの運動学情報。
        ctrl_target 制御対象の情報。
        """
        return self.coating_motion_control(
            control_timestep=control_timestep,
            current_position=kin_state.pos,
            current_quaternion=kin_state.quat,
            current_velocity=kin_state.vel,
            current_ang_velocity=kin_state.ang_vel,
            target_position=ctrl_target.pos,
            target_velocity=ctrl_target.vel,
            target_rpy=ctrl_target.rpy,
            target_rpy_rates=ctrl_target.rpy_rates,
        )

    def coating_attitude_control(
            self,
            control_timestep: float,
            current_position: np.ndarray,
            current_quaternion: np.ndarray,
            current_velocity: np.ndarray,
            current_ang_velocity: np.ndarray,
            target_position: np.ndarray,
            target_velocity: np.ndarray = np.zeros(3),
            target_rpy: np.ndarray = np.zeros(3),
            target_rpy_rates: np.ndarray = np.zeros(3),
    ) -> Tuple:

        cur_rotation = np.array(p.getMatrixFromQuaternion(current_quaternion)).reshape(3, 3) # 現在の回転行列を取得

        gf_thrust = np.array([-0.0981, 0, self._gf]) # ホバリング
        # dot 行列積
        scalar_thrust = max(0, np.dot(gf_thrust, cur_rotation[:, 2])) # 推力をスカラーに変換 cur_rotation[:, 2]は現在の機体のZ軸の向き 
        # 目標推力と現在の機体のZ軸の向きの内積を計算し、その値を0と比較 推力が負の値になることを防ぐ
        thrust = (math.sqrt(scalar_thrust / (4 * self._kf)) - self._pwm2rpm_const) / self._pwm2rpm_scale # 目標推力をドローンの推力 (RPM) に変換

        target_z_ax = gf_thrust / np.linalg.norm(gf_thrust)  # 目標Z軸を計算 目標推力から得られたZ軸の向き（target_z_ax）
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        # target_x_cは、目標のロール、ピッチ、およびヨーに基づいて計算された、指定されたヨー角度に回転された基準のX軸方向の単位ベクトル
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose() # 目標機体の向き（target_rotation）の計算

        # Target rotation 
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False) # 目標回転

        rpm = self.dsl_pid_coating_attitude_control( # 姿勢制御関数の呼び出し
            control_timestep,
            thrust,
            current_quaternion,
            target_euler,
            target_rpy_rates,
        )
        cur_rpy = p.getEulerFromQuaternion(current_quaternion) # 現在のオイラー角を取得
        return rpm

    def coating_motion_control(
            self,
            control_timestep: float,
            current_position: np.ndarray,
            current_quaternion: np.ndarray,
            current_velocity: np.ndarray,
            current_ang_velocity: np.ndarray,
            target_position: np.ndarray,
            target_velocity: np.ndarray,
            target_rpy: np.ndarray,
            target_rpy_rates: np.ndarray = np.zeros(3),
    ) -> Tuple:

         # 速度制御
        thrust, _, _ = self.dsl_pid_position_control(
            control_timestep,
            current_position,
            current_quaternion,
            current_velocity,
            target_position,
            target_velocity,
            target_rpy,
        )
        
        rpm = self.dsl_pid_coating_attitude_control( # 姿勢制御関数の呼び出し
            control_timestep,
            thrust,
            current_quaternion,
            target_rpy,
            target_rpy_rates,
        )
        cur_rpy = p.getEulerFromQuaternion(current_quaternion) # 現在のオイラー角を取得
        return rpm

    def compute_control(
            self,
            control_timestep: float,
            current_position: np.ndarray,
            current_quaternion: np.ndarray,
            current_velocity: np.ndarray,
            current_ang_velocity: np.ndarray,
            target_position: np.ndarray,
            target_velocity: np.ndarray = np.zeros(3),
            target_rpy: np.ndarray = np.zeros(3),
            target_rpy_rates: np.ndarray = np.zeros(3),
    ) -> Tuple:

        # 位置制御関数の呼び出し
        thrust, computed_target_rpy, pos_e = self.dsl_pid_position_control(
            control_timestep,
            current_position,
            current_quaternion,
            current_velocity,
            target_position,
            target_velocity,
            target_rpy,
        )
        rpm = self.dsl_pid_attitude_control( # 姿勢制御関数の呼び出し
            control_timestep,
            thrust,
            current_quaternion,
            computed_target_rpy,
            target_rpy_rates,
        )
        cur_rpy = p.getEulerFromQuaternion(current_quaternion) # 現在のオイラー角を取得
        return rpm, pos_e, computed_target_rpy[2] - cur_rpy[2]

    def dsl_pid_position_control(
            self,
            control_timestep: float,
            current_position: np.ndarray,
            current_quaternion: np.ndarray,
            current_velocity: np.ndarray,
            target_position: np.ndarray,
            target_velocity: np.ndarray,
            target_rpy: np.ndarray,
    ) -> Tuple:
        cur_rotation = np.array(p.getMatrixFromQuaternion(current_quaternion)).reshape(3, 3) # 現在の回転行列を取得
        pos_e = target_position - current_position # 位置エラーと速度エラーを計算
        vel_e = target_velocity - current_velocity

        # 位置制御のpidの積分項
        self._integral_pos_e = self._integral_pos_e + pos_e * control_timestep # 位置エラーの積分項を更新 上で定義 self._integral_pos_e = np.zeros(3)
        self._integral_pos_e = np.clip(self._integral_pos_e, -2., 2.) # clip : 指定された範囲に配列の値をクリップ（制約）
        self._integral_pos_e[2] = np.clip(self._integral_pos_e[2], -0.15, 0.15) # numpy.clip(a, a_min, a_max, out=None, **kwargs)を意味する
        # 位置エラーの3番目の項 clip → 特定の制約内で収束するように調整　安定性の向上

        # PID target thrust # PID制御で求めた目標推力 np.multiply : 乗算掛け算
        # target_thrustが操作量　与えるべきスラストの大きさはいくつか
        target_thrust = np.multiply(self._PID.P_for, pos_e) \
                        + np.multiply(self._PID.I_for, self._integral_pos_e) \
                        + np.multiply(self._PID.D_for, vel_e) \
                        + np.array([0, 0, self._gf])
        scalar_thrust = max(0, np.dot(target_thrust, cur_rotation[:, 2])) # 推力をスカラーに変換 cur_rotation[:, 2]は現在の機体のZ軸の向き 
        # 目標推力と現在の機体のZ軸の向きの内積を計算し、その値を0と比較 推力が負の値になることを防ぐ
        thrust = (math.sqrt(scalar_thrust / (4 * self._kf)) - self._pwm2rpm_const) / self._pwm2rpm_scale # 目標推力をドローンの推力 (RPM) に変換
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)  # 目標Z軸を計算 目標推力から得られたZ軸の向き（target_z_ax）
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        # target_x_cは、目標のロール、ピッチ、およびヨーに基づいて計算された、指定されたヨー角度に回転された基準のX軸方向の単位ベクトル
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose() # 目標機体の向き（target_rotation）の計算

        # Target rotation 
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False) # 目標回転
        # Rotation.from_matrix(target_rotation) は、与えられた回転行列 target_rotation から回転オブジェクトを作成
        # .as_euler('XYZ', degrees=False) は、この回転オブジェクトをXYZオイラー角に変換
        # XYZ' は、変換の順序を指定  ラジアンで角度を表す
        if np.any(np.abs(target_euler) > math.pi): # オイラー角が範囲[-π, π]を超えた場合にログを出力
            logger.error(f"ctrl it {self._env.get_sim_counts()} in {self.__class__.__name__}, range [-pi, pi]")

        return thrust, target_euler, pos_e

    def dsl_pid_attitude_control(
            self,
            control_timestep: float,
            thrust: float,
            current_quaternion: np.ndarray,
            target_euler: np.ndarray,
            target_rpy_rates: np.ndarray,
    ) -> np.ndarray:
        cur_rotation = np.array(p.getMatrixFromQuaternion(current_quaternion)).reshape(3, 3) # 現在の回転行列を取得
        cur_rpy = np.array(p.getEulerFromQuaternion(current_quaternion)) # 現在のオイラー角を取得
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat() # 目標クォータニオンを計算
        w, x, y, z = target_quat # 目標回転行列を計算
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()), cur_rotation) - np.dot(cur_rotation.transpose(), # 回転行列エラーを計算
                                                                                    target_rotation) # rot_matrix_e: 目標回転行列と現在の回転行列のエラー
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) # rot_e: 回転行列エラーから角速度エラーを計算
        rpy_rates_e = target_rpy_rates - (cur_rpy - self._last_rpy) / control_timestep # rpy_rates_e: 目標角速度と現在の角速度のエラーを計算
        self._last_rpy = cur_rpy # 現在のオイラー角を更新
        self._integral_rpy_e = self._integral_rpy_e - rot_e * control_timestep # 角速度エラーの積分項を更新
        self._integral_rpy_e = np.clip(self._integral_rpy_e, -1500., 1500.)
        self._integral_rpy_e[0:2] = np.clip(self._integral_rpy_e[0:2], -1., 1.) # _integral_rpy_e の最初の2つの要素（角速度エラーのxとy成分に対応）
        # PID target torques # PID制御で求めた目標トルク
        target_torques = - np.multiply(self._PID.P_tor, rot_e) \
                         + np.multiply(self._PID.D_tor, rpy_rates_e) \
                         + np.multiply(self._PID.I_tor, self._integral_rpy_e)

        target_torques = np.clip(target_torques, -3200, 3200) # トルクを制約内にクリップ
        pwm = thrust + np.dot(self._Mixer, target_torques) # 目標トルクを、モーターミキシング行列 _Mixer を用いてPWMに変換
        pwm = np.clip(pwm, self._min_pwm, self._max_pwm) # 推力とトルクからPWMを計算
        return self._pwm2rpm_scale * pwm + self._pwm2rpm_const # PWMからRPMに変換して返す returnはRPM

    def dsl_pid_coating_attitude_control(
            self,
            control_timestep: float,
            thrust: float,
            current_quaternion: np.ndarray,
            target_euler: np.ndarray, # target_euler には、ロール、ピッチ、ヨーの目標方向を表す目標オイラー角が含まれてい
            target_rpy_rates: np.ndarray,
    ) -> np.ndarray:
        cur_rotation = np.array(p.getMatrixFromQuaternion(current_quaternion)).reshape(3, 3) # 現在の回転行列を取得
        cur_rpy = np.array(p.getEulerFromQuaternion(current_quaternion)) # 現在のオイラー角を取得
        target_quat = (Rotation.from_euler('XYZ', target_euler, degrees=False)).as_quat() # 目標クォータニオンを計算
        w, x, y, z = target_quat # 目標回転行列を計算
        target_rotation = (Rotation.from_quat([w, x, y, z])).as_matrix()
        rot_matrix_e = np.dot((target_rotation.transpose()), cur_rotation) - np.dot(cur_rotation.transpose(), # 回転行列エラーを計算
                                                                                    target_rotation) # rot_matrix_e: 目標回転行列と現在の回転行列のエラー
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]]) # rot_e: 回転行列エラーから角速度エラーを計算
        rpy_rates_e = target_rpy_rates - (cur_rpy - self._last_rpy) / control_timestep # rpy_rates_e: 目標角速度と現在の角速度のエラーを計算
        self._last_rpy = cur_rpy # 現在のオイラー角を更新
        self._integral_rpy_e = self._integral_rpy_e - rot_e * control_timestep # 角速度エラーの積分項を更新
        self._integral_rpy_e = np.clip(self._integral_rpy_e, -1500., 1500.)
        self._integral_rpy_e[0:2] = np.clip(self._integral_rpy_e[0:2], -1., 1.) # _integral_rpy_e の最初の2つの要素（角速度エラーのxとy成分に対応）
        # PID target torques # PID制御で求めた目標トルク
        target_torques = - np.multiply(self._PID.P_tor, rot_e) \
                         + np.multiply(self._PID.D_tor, rpy_rates_e) \
                         + np.multiply(self._PID.I_tor, self._integral_rpy_e)

        target_torques = np.clip(target_torques, -3200, 3200) # トルクを制約内にクリップ
        pwm = thrust + np.dot(self._Mixer, target_torques) # 目標トルクを、モーターミキシング行列 _Mixer を用いてPWMに変換
        pwm = np.clip(pwm, self._min_pwm, self._max_pwm) # 推力とトルクからPWMを計算
        return self._pwm2rpm_scale * pwm + self._pwm2rpm_const # PWMからRPMに変換して返す returnはRPM

    def dsl_pid_coating_velocity_control(
        self,
        control_timestep: float,
        current_position: np.ndarray,
        current_quaternion: np.ndarray,
        current_velocity: np.ndarray,
        prev_velocity: np.ndarray,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        target_rpy: np.ndarray,
        target_torques: np.ndarray,
        ) -> Tuple:
            cur_rotation = np.array(p.getMatrixFromQuaternion(current_quaternion)).reshape(3, 3) # 現在の回転行列を取得
            vel_e = target_velocity - current_velocity
            
            # 現在の加速度を推定
            current_acceleration = (current_velocity - prev_velocity) / control_timestep
            acc_e = target_acceleration - current_acceleration

            # 速度制御のPID制御を行う部分
            self._integral_vel_e = self._integral_vel_e + vel_e * control_timestep
            self._integral_vel_e = np.clip(self._integral_vel_e, -2., 2.)
            self._integral_vel_e[2] = np.clip(self._integral_vel_e[2], -0.15, 0.15)
            target_thrust = np.multiply(self._PID.P_for, vel_e) \
                            + np.multiply(self._PID.I_for, self._integral_vel_e) \
                            + np.multiply(self._PID.D_for, acc_e)
            scalar_thrust = max(0, np.dot(target_thrust, cur_rotation[:, 2]))
            # 物体が Z 軸方向にどれだけの推力を受けているか
            # max(0, ...) は、計算された推力が負の場合、ゼロに修正します。つまり、負の推力は考慮されず、最小でもゼロの推力が得られる
            thrust = (math.sqrt(scalar_thrust / (4 * self._kf)) - self._pwm2rpm_const) / self._pwm2rpm_scale

            # pwm = thrust + np.dot(self._Mixer, target_torques) # 目標トルクを、モーターミキシング行列 _Mixer を用いてPWMに変換
            # pwm = np.clip(pwm, self._min_pwm, self._max_pwm) # 推力とトルクからPWMを計算

            # 前回の速度を更新
            prev_velocity = current_velocity

            return thrust, prev_velocity


    # def one_2_3_dim_interface(self, thrust): # 入力として与えられた推力 (thrust) から、各モーターに適用されるPWM値を計算するためのもの
    #     """
    #     :param thrust:
    #         Array of floats of length 1, 2, or 4 containing a desired thrust input.
    #     :return: 4つのモーターそれぞれに適用するPWM値（RPMではない）を計算して返しま
    #         (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.
    #     """
    #     dim = len(np.array(thrust))
    #     pwm = np.clip(
    #         (np.sqrt(np.array(thrust) / (self._kf * (4 / dim))) - self._pwm2rpm_const) / self._pwm2rpm_scale,
    #         self._min_pwm,
    #         self._max_pwm,
    #     )
    #     assert dim in [1, 2, 4], f'in one_2_3_dim_interface()'

    #     if dim in [1, 4]:
    #         return np.repeat(pwm, 4 / dim)
    #     elif dim == 2:
    #         return np.hstack([pwm, np.flip(pwm)])

if __name__ == "__main__":

    # ドローンのURDFファイルと初期設定
    # urdf_file = './assets/drone_x_01.urdf'
    # urdf_file = './assets/a.urdf'
    # urdf_file = './assets/b.urdf'
    urdf_file = './assets/b_friction.urdf'
    # urdf_file = './assets/b contact.urdf'
    drone_type = DroneType.QUAD_X # data_definition
    # phy_mode = PhysicsType.PYB_DW # apply_rotor_physics, apply_downwash, pyb_dw
    phy_mode = PhysicsType.PYB
    # phy_mode = PhysicsType.DYN
    # phy_mode = PhysicsType.PYB_DW

    # ドローンの環境を作成
    env = DroneBltEnv( # drone.py
        urdf_path=urdf_file,
        d_type=drone_type,
        is_gui=True,
        phy_mode=phy_mode,
        is_real_time_sim=True,
    )

    # controller ドローンの制御コントローラの初期化
    pid = DroneForcePIDCoefficients( # DroneForcePIDCoefficients : data_definition.py
        # P_for=np.array([.4, .4, 1.25]), # force gain before adjustment
        P_for=np.array([.2, .2, 1.25]),
        # I_for=np.array([.05, .05, .05]), # before adjustment
        # I_for=np.array([.05, .05, .05]),
        I_for=np.array([0, 0, 0]),
        # D_for=np.array([.2, .2, .5]), # before adjustment
        D_for=np.array([.2, .2, .5]),
        # P_tor=np.array([70000., 70000., 60000.]), # torque gain before adjustment
        P_tor=np.array([75000., 75000., 60000.]), # torque gain
        # I_tor=np.array([0., 0., 500.]), # before adjustment
        # I_tor=np.array([1000., 1000., 1000.]),
        I_tor=np.array([250., 250., 500.]),
        D_tor=np.array([20000., 20000., 12000.]), # before adjustment
        # D_tor=np.array([60000., 60000., 36000.]),
    )

    ctrl = DSLPIDControl(env, pid_coeff=pid) # DSLPIDControl : drone_ctrl.py
    # pid_coeff: DroneForcePIDCoefficients, 

    # ドローンの回転数（RPM)設定
    # rpms = np.array([14300, 14300, 14300, 14300])
    rpms = np.array([8000, 8000, 8000, 8000])
    # rpms = np.array([10, 10, 10, 10])

    # # Initial target position 初期位置
    # pos = np.array([-0.2, 0.3, 0.8])

    # # GUIから目標位置を設定するためのパラメータ
    # s_target_x = p.addUserDebugParameter("target_x", -2, 2, pos[0])
    # s_target_y = p.addUserDebugParameter("target_y", -2, 2, pos[1])
    # s_target_z = p.addUserDebugParameter("target_z", 0, 4, pos[2])
    # # p.addUserDebugParameter(name=, rangeMin, rangeMax, startValue)

    # def get_gui_values():
    #     tg_x = p.readUserDebugParameter(int(s_target_x))
    #     tg_y = p.readUserDebugParameter(int(s_target_y))
    #     tg_z = p.readUserDebugParameter(int(s_target_z))
    #     return tg_x, tg_y, tg_z

    # ユーザーが調整可能なパラメータを追加
    yaw_slider = p.addUserDebugParameter("Yaw", -180, 180, 15)
    pitch_slider = p.addUserDebugParameter("Pitch", -90, 90, -30)
    distance_slider = p.addUserDebugParameter("Camera Distance", -5, 10, 0.7) # 0.4
    target_x_slider = p.addUserDebugParameter("Camera TargetPos X", -3, 3, -0.4)
    target_y_slider = p.addUserDebugParameter("Camera TargetPos Y", -3, 3, 0)
    target_z_slider = p.addUserDebugParameter("Camera TargetPos Z", 0, 3, 0.6)
    # p.addUserDebugParameter(name=, rangeMin, rangeMax, startValue)

    def get_camera_values():
        tg_yaw = p.readUserDebugParameter(int(yaw_slider))
        tg_pitch = p.readUserDebugParameter(int(pitch_slider))
        tg_dis = p.readUserDebugParameter(int(distance_slider))
        tg_x_cam = p.readUserDebugParameter(int(target_x_slider))
        tg_y_cam = p.readUserDebugParameter(int(target_y_slider))
        tg_z_cam = p.readUserDebugParameter(int(target_z_slider))
        return tg_yaw, tg_pitch, tg_dis, tg_x_cam, tg_y_cam, tg_z_cam

    # シミュレーションステップの実行
    step_num = 8_000
    # step_num = 2_000
    log_interval = 100  # ログを取る間隔

    target_index = 0  # 現在の目標位置のインデックス

    current_pos = []  # ゼロまたは初期位置外で初期化

    # 前回の接触状態を保存する変数
    previous_contacts = []

    drone_id, wall_id = env.get_drone_inf()

    m, g = env.get_gravity_and_mass()

    print(f"Mass: {m}, Gravity: {g}")
    print(f"mg = : {m * g}")

    #目標移動距離
    d = 0.5

    #目標筆圧
    target_brush_force = 0.0981 # (0.0981N = 0.01kgf)

    # アークサインを計算
    target_angle_rad = math.asin(target_brush_force / (m * g))

    target_quat = np.array([np.cos(target_angle_rad / 2), 0, np.sin(target_angle_rad / 2), 0])

    # クォータニオンからオイラー角に変換
    target_euler = (Rotation.from_quat(target_quat)).as_euler('XYZ', degrees=False)

    # 許容範囲内の位置差
    pos_tolerance = 0.001  # 適切な値に変更してください

    # 許容範囲内の姿勢差
    orientation_tolerance = 0.001  # 適切な値に変更してください

    # 現在の目標位置のインデックス
    current_target_index = 0

    # # 初期の目標位置と次の目標位置を設定
    # target_positions = [
    #     [0, 0, 0.4], 
    #     [-0.2, 0.3, 0.8], 
    #     [current_target_pos[0]-0.1, current_target_pos[1], current_target_pos[2]], 
    #     [current_target_pos[0], current_target_pos[1], current_target_pos[2]], 
    #     [current_target_pos[0], current_target_pos[1], current_target_pos[2]-0.5],
    #     [current_target_pos[0]+0.1, current_target_pos[1], current_target_pos[2]],
    #     [0, 0, 0], 
    #     ]  # 例として初期位置とその次の位置を設定

    # target_positions = [
    #     [0, 0, 0.4], 
    #     [-0.1, 0.3, 0.8], 
    #     [-0.3, 0.3, 0.8], 
    #     [-0.3, 0.3, 0.8], 
    #     [-0.3, 0.3, 0.3], 
    #     [-0.2, 0.3, 0.3], 
    #     [0, 0, 0], 
    #     ]  # 初期位置とその次の位置を設定

    target_positions = [
        [0, 0, 0.4], 
        [-0.3, 0.3, 0.8], 
        [0, 0, 0],
        ]

    # target_velocity = [
    #     [0, 0, 0], 
    #     [0, 0, -0.5], 
    #     ]  # 初期位置とその次の位置を設定

    # t = 3.0

    # target_rpys = [
    #     [0, 0, 0],
    #     [0, 0, 0],
    #     [0, 0, 0],
    #     [0, target_angle_rad, 0],
    #     [0, target_angle_rad, 0],
    #     [0, 0, 0],
    #     [0, 0, 0],
    #     ]  # 例として初期位置とその次の位置を設定

    target_rpys = [
        [0, 0, 0],
        [0, 0, 0],
        [0, target_angle_rad, 0],
        ]  # 初期位置とその次の位置を設定

    current_target_pos = target_positions[0]

    current_target_rpy = target_rpys[0]

    pos_difference = [float('inf')] * len(target_positions)

    orientation_difference = [float('inf')] * len(target_rpys)

    prev_velocity = np.array([0, 0, 0,])

    flag_condition0 = False
    flag_condition1 = False
    flag_condition2 = False
    flag_condition3 = False
    flag_condition4 = False
    flag_condition5 = False
    flag_condition6 = False

    for i in range(step_num):
        kis = env.step(rpms) # drone.py stepメソッド　ドローンを指定された回転速度で1ステップ進める
        # current_position = kis[0].pos

        # カメラの目標位置を取得
        tg_yaw, tg_pitch, tg_dis, tg_x_cam, tg_y_cam, tg_z_cam = get_camera_values()

        # カメラの情報を設定
        p.resetDebugVisualizerCamera(cameraDistance=tg_dis,
                                    cameraYaw=tg_yaw,
                                    cameraPitch=tg_pitch,
                                    cameraTargetPosition=[tg_x_cam, tg_y_cam, tg_z_cam])

        # 物体の現在の位置を取得
        current_pos, current_quat = p.getBasePositionAndOrientation(drone_id)

        current_rpy = p.getEulerFromQuaternion(current_quat)

        # 目標位置に到達したかどうかを判断
        pos_difference[current_target_index] = np.linalg.norm(np.array(current_pos) - np.array(current_target_pos))

        # 現在の目標姿勢に到達したかどうかを判断
        orientation_difference[current_target_index] = np.linalg.norm(np.array(current_rpy) - np.array(target_rpys[current_target_index]))

        # # 現在の目標位置
        # current_target_pos = target_positions[current_target_index]

        # current_target_rpy = target_rpys[current_target_index]

        # 接触情報の取得
        contacts = p.getContactPoints(bodyA=drone_id, bodyB=wall_id)

        # while current_position != init_xyzs:
        # 制御アルゴリズムに基づいて次の回転数（RPM）を計算
        # 与えられた入力に基づいて次の回転数（RPM）を計算し、それをrpmsに格納

        if current_target_index < 2 :
            rpms, _, _ = ctrl.compute_control_from_kinematics( # drone_ctrl.pyのメソッド
                control_timestep=env.get_sim_time_step(), # control_timestep: 制御のタイムステップ（時間間隔）　# シミュレーションの時間ステップを取得
                kin_state=kis, # kin_state: 現在のドローンの運動学的状態（姿勢や速度など）# kin_state: DroneKinematicsInfo : 情報の格納場所　kis[0]　ドローンが一体
                ctrl_target=DroneControlTarget( # ctrl_target: 制御の目標となる値。DroneControlTargetクラスのインスタンスで、目標位置が指定されている
                    pos=current_target_pos, # DroneControlTarget : data_difinition.py 
                ), # DroneControlTargetのposに格納
            )

        else:
            rpms = ctrl.coating_attitude_control_from_kinematics( # drone_ctrl.pyのメソッド
                control_timestep=env.get_sim_time_step(), # control_timestep: 制御のタイムステップ（時間間隔）　# シミュレーションの時間ステップを取得
                kin_state=kis, # kin_state: 現在のドローンの運動学的状態（姿勢や速度など）# kin_state: DroneKinematicsInfo : 情報の格納場所　kis[0]　ドローンが一体
                ctrl_target=DroneControlTarget( # ctrl_target: 制御の目標となる値。DroneControlTargetクラスのインスタンスで、目標位置が指定されている
                    rpy=target_euler,
                ), # DroneControlTargetのposに格納
            )

        # # 接触情報の取得
        # contacts = p.getContactPoints(bodyA=drone_id, bodyB=wall_id)

        if i % log_interval == 0: # i が log_interval の倍数のときに条件が真になり、そのときにログが出力
            # i % log_intervalはi を log_interval で割った余り　それが0になる時
            # 500ステップごとにrpyとxyzの値を出力
            print(f"Step {i}:")
            print("Roll, Pitch, Yaw:", kis.rpy)
            print("Position (XYZ):", kis.pos)
            print("Velocity (XYZ):", kis.vel)
            # print(f"Force on {target_link_name}: {force_on_link}") # [N]ニュートン
            # print(f"Torque on {target_link_name}: {torque_on_link}")
            if contacts:
                print("棒と壁が接触しています")
                # 接触点ごとに情報を表示
                for contact_info in contacts:
                    # print(f"  Position on stick: {contact_info['positionOnA']}")
                    # print(f"  Position on wall: {contact_info['positionOnB']}")
                    print(f"  Normal force: {contact_info[9]}") # 接触点での法線方向（垂直方向）への力
                    # 法線方向の力がプラスの場合、物体同士が互いに押し合っていることを示す
                    # print(f"  lateralFriction1: {contact_info['lateralFriction1']}")
                    # print(f"  lateralFrictionDir1: {contact_info['lateralFrictionDir1']}")
                    # print(f"  lateralFriction2: {contact_info['lateralFriction2']}")
                    brush_force = contact_info[9]

                    # アークサインを計算
                    angle_rad = math.asin(brush_force / (m * g))

                    angle_deg = math.degrees(angle_rad)
                    # 結果を表示
                    print(f"アークサインは {angle_rad} ラジアンです。")
                    print(f"アークサインは {angle_deg} 度です。")

        if pos_difference[0] < pos_tolerance and not flag_condition0:
            print(f"目標位置 {current_target_index} に到達しました。")
            print(f"目標姿勢 {current_target_index} に到達しました。")

            # 次の目標位置のインデックスを更新
            current_target_index += 1

            current_target_pos = target_positions[current_target_index]

            # フラグをセット
            flag_condition0 = True

        if pos_difference[1] < pos_tolerance and not flag_condition1:
            print(pos_difference[1])
            print(f"目標位置 {current_target_index} に到達しました。")
            print(f"目標姿勢 {current_target_index} に到達しました。")

            # 次の目標位置のインデックスを更新
            current_target_index += 1

            current_target_pos = target_positions[current_target_index]

            # フラグをセット
            flag_condition1 = True

        # if  orientation_difference[2] < orientation_tolerance :
        #     print(f"目標位置 {current_target_index} に到達しました。")
        #     print(f"目標姿勢 {current_target_index} に到達しました。")
        #     print("Current Position:", current_pos)

        #     # 次の目標位置のインデックスを更新
        #     current_target_index += 1

        #     # フラグをセット
        #     flag_condition2 = True

        # if pos_difference[3] < pos_tolerance and orientation_difference[3] < orientation_tolerance and not flag_condition3:
        #     print(f"目標位置 {current_target_index} に到達しました。")
        #     print(f"目標姿勢 {current_target_index} に到達しました。")

        #     # 次の目標位置のインデックスを更新
        #     current_target_index += 1

        #     # フラグをセット
        #     flag_condition3 = True

        # if pos_difference[4] < pos_tolerance and orientation_difference[4] < orientation_tolerance and not flag_condition4:
        #     print(f"目標位置 {current_target_index} に到達しました。")
        #     print(f"目標姿勢 {current_target_index} に到達しました。")

        #     # 次の目標位置のインデックスを更新
        #     current_target_index += 1

        #     # フラグをセット
        #     flag_condition4 = True

        # if pos_difference[5] < pos_tolerance and orientation_difference[5] < orientation_tolerance and not flag_condition5:
        #     print(f"目標位置 {current_target_index} に到達しました。")
        #     print(f"目標姿勢 {current_target_index} に到達しました。")

        #     # 次の目標位置のインデックスを更新
        #     current_target_index += 1

        #     # フラグをセット
        #     flag_condition5 = True

        # if pos_difference[6] < pos_tolerance and orientation_difference[6] < orientation_tolerance and not flag_condition6:
        #     print(f"目標位置 {current_target_index} に到達しました。")
        #     print(f"目標姿勢 {current_target_index} に到達しました。")

        #     # 次の目標位置のインデックスを更新
        #     current_target_index += 1

        #     # フラグをセット
        #     flag_condition6 = True                       

        # 全ての目標位置に到達した場合、ループを抜ける
        if current_target_index >= len(target_rpys):
            print("全ての目標位置に到達しました。")
            break
        # # 前回のステップで接触しておらず、今回のステップで接触した場合
        # if not previous_contacts and contacts:

        #     print("Current Position:", current_pos)

        #     # 現在の目標位置
        #     current_target_pos = np.array([current_pos[0], current_pos[1], current_pos[2] - d])
        #     current_target_rpy = np.array([0.0, target_angle_rad, 0.0])

        # if position_difference < position_tolerance:
        #     print("目標位置に到達しました。")

        #     print("Current Position:", current_pos)

        #     # 現在の目標位置
        #     current_target_pos = np.array([current_pos[0] + 0.1, current_pos[1], current_pos[2]])
        #     current_target_rpy = init_rpy

        #     break  # 目標位置に到達したらループを抜ける
        
        # if position_difference < position_tolerance:
        #     print("目標位置に到達しました。")

        #     print("Current Position:", position)

        #     # 現在の目標位置
        #     current_target_pos = init_xyzs
        #     current_target_rpy = init_rpy

        #     break  # 目標位置に到達したらループを抜ける



        # # ここで次の目標位置に移動するための処理を追加

        # # 前回のステップで接触しておらず、今回のステップで接触した場合
        # if not previous_contacts and contacts:

        #     print("Current Position:", current_pos)

        #     # 現在の目標位置
        #     current_target_pos = np.array([current_pos[0], current_pos[1], current_pos[2] - d])
        #     current_target_rpy = np.array([0.0, target_angle_rad, 0.0])
        

        # # 今回の接触情報を保存
        # previous_contacts = contacts

            # pitchを変える

            # 初期位置への移動
            # rpms, _, _ = ctrl.compute_control_from_kinematics(
            #     control_timestep=env.get_sim_time_step(),
            #     kin_state=kis,
            #     ctrl_target=DroneControlTarget(
            #         pos=np.array([init_xyzs[0], init_xyzs[1], init_xyzs[2]]),
            #     ),
            # )

            # # 目標の姿勢と位置
            # target_position_after_descent = np.array([target_x_after_descent, target_y_after_descent, target_z_after_descent])
            # target_orientation_after_descent = np.array([target_roll_after_descent, target_pitch_after_descent, target_yaw_after_descent])

            # # カメラの目標位置を変更
            # # ... (省略)

            # # ロール、ピッチ、ヨー、位置、速度をログに出力
            # # ... (省略)

            # # 接触情報の取得
            # contacts = p.getContactPoints(bodyA=drone_id, bodyB=wall_id)

            # if contacts:
            #     for contact_info in contacts:
            #         brush_force = contact_info[9]
            #         angle_rad = math.asin(brush_force / (m * g))
            #         angle_deg = math.degrees(angle_rad)
            #         print(f"アークサインは {angle_rad} ラジアンです。")
            #         print(f"アークサインは {angle_deg} 度です。")

            #         # 目標位置と姿勢の変更
            #         rpms, _, _ = ctrl.compute_control_from_kinematics(
            #             control_timestep=env.get_sim_time_step(),
            #             kin_state=kis,
            #             ctrl_target=DroneControlTarget(
            #                 pos=target_position_after_descent,
            #                 rpy=target_orientation_after_descent,
            #             ),
            #         )

            # # 一定の下降距離まで下降
            # # ... (省略)

            # # 元の姿勢に戻る
            # rpms, _, _ = ctrl.compute_control_from_kinematics(
            #     control_timestep=env.get_sim_time_step(),
            #     kin_state=kis,
            #     ctrl_target=DroneControlTarget(
            #         pos=np.array([init_xyzs[0], init_xyzs[1], init_xyzs[2]]),
            #         rpy=np.array([init_roll, init_pitch, init_yaw]),
            #     ),
            # )


        # # 接触しているかどうかを判定
        # if contacts:
        #     print("棒と壁が接触しています")
        #     rpms, _, _ = ctrl.compute_control_from_kinematics( # drone_ctrl.pyのメソッド
        #     control_timestep=env.get_sim_time_step(), # control_timestep: 制御のタイムステップ（時間間隔）　# シミュレーションの時間ステップを取得
        #     kin_state=kis, # kin_state: 現在のドローンの運動学的状態（姿勢や速度など）# kin_state: DroneKinematicsInfo : 情報の格納場所　kis[0]　ドローンが一体
        #     ctrl_target=DroneControlTarget( # ctrl_target: 制御の目標となる値。DroneControlTargetクラスのインスタンスで、目標位置が指定されている
        #         pos=np.array([tg_x, tg_y, tg_z]), # DroneControlTarget : data_difinition.py 
        #         rpy=np.array([tg_x, tg_y, tg_z])
        #     ), # DroneControlTargetのposに格納
        # )

        
        # # Log the simulation (optional).
        # t_stamp = i / env.get_sim_freq()
        # d_log.log(
        #     drone_id=0,
        #     time_stamp=t_stamp,
        #     kin_state=kis[0],
        # )

    # Close the environment
    env.close()

    # # Plot the simulation results (optional).
    # d_log.plot()


# 観測されている6つの値（Roll、Pitch、Yaw、Position（XYZ）、Velocity（XYZ））は、おそらくコード内のprintステートメントの出力