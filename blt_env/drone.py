from typing import Optional, List, Tuple, Union
import time
import random

from logging import getLogger, NullHandler

import numpy as np
import pybullet as p
import pybullet_data

from blt_env.bullet_base import BulletEnv

from util.data_definition import DroneProperties, DroneType, DroneKinematicsInfo, PhysicsType
from util.file_tools import DroneUrdfAnalyzer

# ロギングのためのインスタンスを生成
# loggerという変数が新しいロギング（ログ出力）のためのインスタンスを生成している
# このインスタンスを通じて、プログラムの実行中にログメッセージを出力できる
logger = getLogger(__name__) 
logger.addHandler(NullHandler())


# logger.setLevel(DEBUG)  # for standalone debugging
# logger.addHandler(StreamHandler())  # for standalone debugging

def real_time_step_synchronization(sim_counts, start_time, time_step):
    """Syncs the stepped simulation with the wall-clock.

    This is a reference from the following ...

        https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/utils/utils.py

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    sim_counts : int
        Current simulation iteration. # 現在のシミュレーションイテレーション。
    start_time : timestamp   シミュレーション開始時のタイムスタンプ。
        Timestamp of the simulation start.
    time_step : float シミュレーションのレンダリングにおける所望の壁時計（現実の時間）ステップ。
        Desired, wall-clock step of the simulation's rendering.

    """
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
            num_drones: int = 1,
            is_gui: bool = True,
            is_real_time_sim: bool = False,
            init_xyzs: Optional[Union[List, np.ndarray]] = None,
            init_rpys: Optional[Union[List, np.ndarray]] = None,
    ):
        """
        'aggr_phy_steps'は、self.step(rpm_values: np.ndarray)が１回よばれた際に、PyBulletによるシミュレーションを
        何ステップ実行するかを指定する。この値が増加すると、actionの頻度は減少する。
        （この場合のactionとは、rpm_valuesをself.step()に引数として与えてドローンをコントロールすること）

        Parameters
        ----------
        urdf_path : The drone *.URDF file path.
        d_type : Specifies the type of drone to be loaded from the *.URDF file.
        phy_mode : Specifies the type of physics simulation for PyBullet.
        sim_freq : Specifies the frequency of the PyBullet step simulations.
        aggr_phy_steps : The number of physics steps within one call to `self.step()`.
                        The frequency of the control action is changed by the aggr_phy_steps.
        num_drones : Number of drones to be loaded.
        is_gui : Whether to start PyBullet in GUI mode.
        """
        super().__init__(is_gui=is_gui)
        self._drone_type = d_type
        self._urdf_path = urdf_path
        self._physics_mode = phy_mode

        self._dp = load_drone_properties(self._urdf_path, self._drone_type)
        # self.printout_drone_properties()

        # PyBullet simulation settings. PyBulletシミュレーション設定
        self._num_drones = num_drones # ドローンの数
        self._aggr_phy_steps = aggr_phy_steps # 物理シミュレーションのステップ数
        self._g = self._dp.g # 重力の値
        self._sim_freq = sim_freq # PyBulletステップシミュレーションの頻度
        self._sim_time_step = 1. / self._sim_freq # シミュレーションのタイムステップ
        self._is_realtime_sim = is_real_time_sim  # add wait time in step(). # step()において待機時間を追加するかどうか

        # Initialization position of the drones. ドローンの初期位置
        if init_xyzs is None:
            self._init_xyzs = np.vstack([ # np.vstack 配列の結合　行が増える
                np.array([x * 4 * self._dp.l for x in range(self._num_drones)]), # X座標
                np.array([y * 4 * self._dp.l for y in range(self._num_drones)]), # Y座標
                np.ones(self._num_drones) * (self._dp.collision_h / 2 - self._dp.collision_z_offset + 0.1), # np.ones 全ての要素が1の配列
            ]).transpose().reshape(self._num_drones, 3) # Z座標
        else: # もし初期位置が指定されている場合
            assert init_xyzs.ndim == 2, f"'init_xyzs' should has 2 dimension. current dims are {init_xyzs.ndim}."
            self._init_xyzs = np.array(init_xyzs) # 初期位置を指定された値にセット
        assert self._init_xyzs.shape[0] == self._num_drones, f""" Initialize position error. # 初期位置の行数がドローンの数と一致していることを確認
        Number of init pos {self._init_xyzs.shape[0]} vs number of drones {self._num_drones}."""

        if init_rpys is None: # もし初期姿勢が指定されていない場合は、ゼロ行列で初期化
            self._init_rpys = np.zeros((self._num_drones, 3))
        else: # もし初期姿勢が指定されている場合
            assert init_rpys.ndim == 2, f"'init_rpys' should has 2 dimension. current dims are {init_rpys.ndim}."
            self._init_rpys = np.array(init_rpys)  # 初期姿勢を指定された値にセット
        assert self._init_rpys.shape[0] == self._num_drones, f""" Initialize roll, pitch and yaw error.
        Number of init rpy {self._init_rpys.shape[0]} vs number of drones {self._num_drones}.""" # 初期姿勢の行数がドローンの数と一致していることを確認

        # Simulation status. シミュレーションの状態
        self._sim_counts = 0 # シミュレーションの実行回数
        self._last_rpm_values = np.zeros((self._num_drones, 4)) # 直前のrpm_values（ドローンの回転速度）の初期化
        ''' 
        The 'DroneKinematicInfo' class is simply a placeholder for the following information.
            pos : position
            quat : quaternion
            rpy : roll, pitch and yaw
            vel : linear velocity
            ang_vel : angular velocity
        '''
        self._kis = [DroneKinematicsInfo() for _ in range(self._num_drones)]

        # もし物理モードがDYN（動力学モード）の場合は、角速度を保持する変数を初期化
        if self._physics_mode == PhysicsType.DYN: 
            self._rpy_rates = np.zeros((self._num_drones, 3))

        # PyBullet environment. PyBullet環境
        self._client = p.connect(p.GUI) if self._is_gui else p.connect(p.DIRECT)
        p.setGravity(0, 0, -self._g, physicsClientId=self._client)
        p.setRealTimeSimulation(0, physicsClientId=self._client)
        p.setTimeStep(self._sim_time_step, physicsClientId=self._client)

        # Load objects. オブジェクトの読み込み
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = p.loadURDF('plane.urdf')

        wall_start_pos = [-0.5, 1, 0.5]
        wall_id = p.loadURDF('./assets/wall.urdf', basePosition=wall_start_pos)

        # Load drones. 各ドローンの読み込み
        self._drone_ids = np.array([
            p.loadURDF(
                self._urdf_path, # URDFファイルのパス
                self._init_xyzs[i, :], # ドローンの初期位置
                p.getQuaternionFromEuler(self._init_rpys[i, :]), # ドローンの初期姿勢（オイラー角からクォータニオンに変換）
            ) for i in range(self._num_drones)])

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

    def refresh_bullet_env(self):
        """
        Refresh the PyBullet simulation environment.
        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `self.reset()` function.

        PyBulletのシミュレーション環境を初期化する

        """
        self._sim_counts = 0 # シミュレーション反復回数をリセット
        self._last_rpm_values = np.zeros((self._num_drones, 4)) # 最後のRPM値をゼロに初期化
        self._kis = [DroneKinematicsInfo() for _ in range(self._num_drones)] # ドローンの運動情報を空のリストで初期化
        if self._physics_mode == PhysicsType.DYN: # 物理モードがDYNの場合、角速度をゼロに初期化
            self._rpy_rates = np.zeros((self._num_drones, 3))

        # Set PyBullet's parameters. PyBulletのパラメータを設定
        p.setGravity(0, 0, -self._g, physicsClientId=self._client)
        p.setRealTimeSimulation(0, physicsClientId=self._client)
        p.setTimeStep(self._sim_time_step, physicsClientId=self._client)

        # Load objects. オブジェクトのロード
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        self._plane_id = p.loadURDF('plane.urdf') # 平面のIDを取得

        # Load drones. ドローンのロード
        self._drone_ids = np.array([
            p.loadURDF(
                self._urdf_path,
                self._init_xyzs[i, :],
                p.getQuaternionFromEuler(self._init_rpys[i, :]),
            ) for i in range(self._num_drones)])

        self.update_drones_kinematic_info()  # ドローンの運動情報を更新

        # Reset measuring time. 時間の計測をリセット
        self._start_time = time.time()

    def update_drones_kinematic_info(self):
        for i in range(self._num_drones):
            pos, quat = p.getBasePositionAndOrientation( # ドローンの位置と姿勢を取得
                bodyUniqueId=self._drone_ids[i],
                physicsClientId=self._client,
            )
            rpy = p.getEulerFromQuaternion(quat) # クォータニオンからロールピッチヨー角を取得
            vel, ang_vel = p.getBaseVelocity( # ドローンの速度と角速度を取得
                bodyUniqueId=self._drone_ids[i],
                physicsClientId=self._client,
            )
            self._kis[i] = DroneKinematicsInfo( # ドローンの運動情報を更新
                pos=np.array(pos),
                quat=np.array(quat),
                rpy=np.array(rpy),
                vel=np.array(vel),
                ang_vel=np.array(ang_vel),
            )

    def close(self) -> None: 
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
            for i in range(self._num_drones):
                self.physics(
                    rpm_values[i, :],
                    i,
                    self._last_rpm_values[i, :],
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

    def check_values_for_rotors(self, rpm_values: np.ndarray) -> np.ndarray:
        """
        Check that 'rpm_values', which specifies the rotation speed of the 4-rotors, are in the proper form.
        Also, if possible, modify 'rpm_values' to the appropriate form.

        各ドローンの4つのロータ回転数を指定するnp.ndarrayが、適切な形式になっているか確認する

        Parameters and Returns
        ----------
        rpm_values : Multiple arrays with 4 values as a pair of element.
                    Specify the rotational speed of the four rotors of each drone.
                    4つの値からなる複数の配列のペア。
                    各ドローンの４つのロータの回転速度を指定する。
        """
        cls_name = self.__class__.__name__
        assert isinstance(rpm_values, np.ndarray), f"Invalid rpm_values type is used on {cls_name}." # f"{cls_name}で無効なrpm_valuesの型が使用されています。
        assert rpm_values.ndim == 1 or rpm_values.ndim == 2, f"Invalid dimension of rpm_values is used on {cls_name}." # f"{cls_name}で無効なrpm_valuesの次元が使用されています。"
        if rpm_values.ndim == 1:
            assert len(rpm_values) == 4, f"Invalid number of elements were used for rpm_values on {cls_name}." # f"{cls_name}で無効なrpm_valuesの要素数が使用されています。"
            ''' e.g. 例
            while, a = [100, 200, 300, 400]
            then, np.tile(a, (3, 1)) -> [[100, 200, 300, 400], [100, 200, 300, 400], [100, 200, 300, 400]]
            '''
            rpm_values = np.tile(rpm_values, (self._num_drones, 1))
        elif rpm_values.ndim == 2:
            assert rpm_values.shape[1] == 4, f"Invalid number of elements were used for rpm_values on {cls_name}." # f"{cls_name}で無効なrpm_valuesの要素数が使用されています。"
            rpm_values = np.reshape(rpm_values, (self._num_drones, 4))
        return rpm_values

    def physics(
            self,
            rpm: np.ndarray,
            nth_drone: int,
            last_rpm: Optional[np.ndarray],
    ) -> None:
        """
        The type of physics simulation will be selected according to 'self._physics_mode'.
        self._physics_mode'に基づいて物理演算モデルが選択されます。

        'self._physics_mode'で指定されたモードにしたがって物理演算モデルが選択される

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        last_rpm : Previous specified value.
        rpm : 4つの要素からなる配列。各ドローンの4つのロータの回転速度を指定します。
        nth_drone : ドローンのリストself._drone_idsでの所定のドローンの序数。
        last_rpm : 前回指定された値。
        """

        def pyb(rpm, nth_drone: int, last_rpm=None):
            self.apply_rotor_physics(rpm, nth_drone)

        def dyn(rpm, nth_drone: int, last_rpm=None):
            self.apply_dynamics(rpm, nth_drone)

        def pyb_gnd(rpm, nth_drone: int, last_rpm=None):
            self.apply_rotor_physics(rpm, nth_drone)
            self.apply_ground_effect(rpm, nth_drone)

        def pyb_drag(rpm, nth_drone: int, last_rpm):
            self.apply_rotor_physics(rpm, nth_drone)
            self.apply_drag(last_rpm, nth_drone)  # apply last data

        def pyb_dw(rpm, nth_drone: int, last_rpm=None):
            self.apply_rotor_physics(rpm, nth_drone)
            self.apply_downwash(nth_drone)

        def pyb_gnd_drag_dw(rpm, nth_drone: int, last_rpm):
            self.apply_rotor_physics(rpm, nth_drone)
            self.apply_ground_effect(rpm, nth_drone)
            self.apply_drag(last_rpm, nth_drone)  # apply last data
            self.apply_downwash(nth_drone)

        def other(rpm, nth_drone: int, last_rpm):
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
        return key_dict.get(phy_key, other)(rpm, nth_drone, last_rpm)

    def apply_rotor_physics(self, rpm: np.ndarray, nth_drone: int):
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

        # 各ローターに推力を適用
        for i in range(4):
            p.applyExternalForce(
                objectUniqueId=self._drone_ids[nth_drone],
                linkIndex=i,  # link id of the rotors.　ローターのリンクID。
                forceObj=[0, 0, forces[i]],
                posObj=[0, 0, 0],
                flags=p.LINK_FRAME,
                physicsClientId=self._client,
            )
        p.applyExternalTorque(
            objectUniqueId=self._drone_ids[nth_drone],
            linkIndex=4,  # link id of the center of mass.　重心のリンクID
            torqueObj=[0, 0, z_torque],
            flags=p.LINK_FRAME,
            physicsClientId=self._client,
        )

    def apply_ground_effect(self, rpm: np.ndarray, nth_drone: int):
        """
        Apply ground effect.
        地面効果を適用

        This is a reference from the following ...

            https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py

            Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        rpm : ドローンの各ローターの回転速度を指定する4つの要素を持つ配列。
        nth_drone : self._drone_ids内で望むドローンの序数。
        """
        assert len(rpm) == 4, f"The length of rpm_values must be 4. currently it is {len(rpm)}."

        ''' getLinkState()
        computeLinkVelocity : 
            If set to 1, the Cartesian world velocity will be computed and returned.
        computeForwardKinematics : 
            If set to 1 (or True), the Cartesian world position/orientation will be recomputed using forward kinematics.
        '''
        # ドローンの各リンクの状態を取得
        link_states = np.array(
            p.getLinkStates(
                bodyUniqueId=self._drone_ids[nth_drone],
                linkIndices=[0, 1, 2, 3, 4],
                computeLinkVelocity=1,
                computeForwardKinematics=1,
                physicsClientId=self._client,
            ),
            dtype=object,
        )

        # Simple, per-propeller ground effects.　単純なプロペラごとの地面効果。
        # 地面効果（Ground Effect）は、航空機が地表に近い高度で飛行する際に発生する気象効果の一つです。
        # これは、地表と航空機の間に空気が押しつぶされることによって、揚力が増加する現象
        # 各プロペラの高さをクリップして計算
        prop_heights = np.array(
            [link_states[0, 0][2], link_states[1, 0][2], link_states[2, 0][2], link_states[3, 0][2]])
        prop_heights = np.clip(prop_heights, self._dp.grand_eff_h_clip, np.inf)
        gnd_effects = np.array(rpm) ** 2 * self._dp.kf * self._dp.gnd_eff_coeff * ( # 地面効果を計算
                self._dp.prop_radius / (4 * prop_heights)) ** 2

        ki = self._kis[nth_drone] # ドローンが水平であるときにのみ地面効果を適用
        if np.abs(ki.rpy[0]) < np.pi / 2 and np.abs(ki.rpy[1]) < np.pi / 2:
            for i in range(4):
                p.applyExternalForce( # 各プロペラに地面効果を適用
                    objectUniqueId=self._drone_ids[nth_drone],
                    linkIndex=i,
                    forceObj=[0, 0, gnd_effects[i]],
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME,
                    physicsClientId=self._client,
                )

    def apply_drag(self, rpm: np.ndarray, nth_drone: int):
        """
        Apply drag force.
        抗力を適用

        This is a reference from the following ...

            https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py

            Based on the the system identification in (Forster, 2015).

            Chapter 4 Drag Coefficients
            http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        """

        # Rotation matrix of the base.　ベースの回転行列を取得
        ki = self._kis[nth_drone]
        base_rot = np.array(p.getMatrixFromQuaternion(ki.quat)).reshape(3, 3)
        # Simple draft model applied to the center of mass　シンプルなドラフトモデルを重心に適用
        drag_factors = -1 * self._dp.drag_coeff * np.sum(2 * np.pi * np.array(rpm) / 60)
        drag = np.dot(base_rot, drag_factors * np.array(ki.vel))
        p.applyExternalForce( # ドラッグ力をドローンの中心に外部力として適用
            objectUniqueId=self._drone_ids[nth_drone],
            linkIndex=4,  # link id of the center of mass. 重心のリンクID
            forceObj=drag,
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=self._client,
        )

    def apply_downwash(self, nth_drone: int):
        """
        Apply downwash.
        ダウンウオッシュ（吹き下ろし）を適用

        The aerodynamic caused by the motion of the rotor blade's airfoil during the process of generating lift.
        Interactions between multiple drones.

        This is a reference from the following ...

            https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py

            Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Parameters
        ----------
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        """
        ki_d = self._kis[nth_drone]
        for i in range(self._num_drones):
            ki_i = self._kis[i]
            delta_z = ki_i.pos[2] - ki_d.pos[2]
            delta_xy = np.linalg.norm(np.array(ki_i.pos[0:2]) - np.array(ki_d.pos[0:2]))
            if delta_z > 0 and delta_xy < 10:  # Ignore drones more than 10 meters away
                alpha = self._dp.dw_coeff_1 * (self._dp.prop_radius / (4 * delta_z)) ** 2
                beta = self._dp.dw_coeff_2 * delta_z + self._dp.dw_coeff_3
                downwash = [0, 0, -alpha * np.exp(-0.5 * (delta_xy / beta) ** 2)]
                p.applyExternalForce(
                    objectUniqueId=self._drone_ids[nth_drone],
                    linkIndex=4,  # link id of the center of mass.
                    forceObj=downwash,
                    posObj=[0, 0, 0],
                    flags=p.LINK_FRAME,
                    physicsClientId=self._client,
                )

    def apply_dynamics(self, rpm: np.ndarray, nth_drone: int):
        """
        Apply dynamics taking into account moment of inertia, etc. (not pybullet base)
        力学を考慮して動力を適用します。慣性モーメントなども考慮されます（PyBulletベースではない）。
        慣性モーメントなどを考慮した力学を陽解法を用いて適用

        This is a reference from the following ...

            https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py

            Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        nth_drone : The ordinal number of the desired drone in list self._drone_ids.
        rpm : 4つの要素からなる配列。各ドローンの4つのローターの回転速度を指定します。
        nth_drone : self._drone_ids内での対象のドローンの番号。
        """
        assert len(rpm) == 4, f"The length of rpm_values must be 4. currently it is {len(rpm)}." # f"rpm_valuesの長さは4でなければなりません。現在の長さは {len(rpm)} です。"

        # Current state. 現在の状態を取得
        ki = self._kis[nth_drone]
        pos = ki.pos
        quat = ki.quat
        rpy = ki.rpy
        vel = ki.vel
        rpy_rates = self._rpy_rates[nth_drone]  # angular velocity
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

        # Compute thrust and torques. 推力とトルクを計算
        thrust, x_torque, y_torque, z_torque = self.rpm2forces(rpm)
        thrust = np.array([0, 0, thrust])

        thrust_world_frame = np.dot(rotation, thrust)
        forces_world_frame = thrust_world_frame - np.array([0, 0, self._dp.gf])

        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self._dp.J, rpy_rates))
        rpy_rates_deriv = np.dot(self._dp.J_inv, torques)  # angular acceleration 角加速度
        no_pybullet_dyn_accs = forces_world_frame / self._dp.m

        # Update state. 状態を更新
        vel = vel + self._sim_time_step * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self._sim_time_step * rpy_rates_deriv
        pos = pos + self._sim_time_step * vel
        rpy = rpy + self._sim_time_step * rpy_rates

        # Set PyBullet state PyBulletの状態を設定
        p.resetBasePositionAndOrientation(
            bodyUniqueId=self._drone_ids[nth_drone],
            posObj=pos,
            ornObj=p.getQuaternionFromEuler(rpy),
            physicsClientId=self._client,
        )

        # Note: the base's velocity only stored and not used. 注: ベースの速度は保存されるだけで使用されません。
        p.resetBaseVelocity(
            objectUniqueId=self._drone_ids[nth_drone],
            linearVelocity=vel,
            angularVelocity=[-1, -1, -1],  # ang_vel not computed by DYN # ang_vel は DYN で計算されない
            physicsClientId=self._client,
        )

        # Store the roll, pitch, yaw rates for the next step 次のステップのためにロール、ピッチ、ヨーレートを保存
        # ki.rpy_rates = rpy_rates 
        self._rpy_rates[nth_drone] = rpy_rates

    def rpm2forces(self, rpm: np.ndarray) -> Tuple:
        """
        Compute thrust and x, y, z axis torque at specified rotor speed.
        指定されたローター速度で推力およびx、y、z軸トルクを計算します。

        Parameters
        ----------
        rpm : A array with 4 elements. Specify the rotational speed of the four rotors of each drone.
        rpm : 4つの要素からなる配列。各ドローンの4つのローターの回転速度を指定します。

        Returns
        -------
        (
            thrust,  # It is sum of the thrust of the 4 rotors.
            x_torque,  # It is the torque generated by the thrust of the rotors.
            y_torque,  # It is the torque generated by the thrust of the rotors.
            z_torque,  #  It is sum of the torque of the 4 rotors.
            thrust,  # 4つのローターの推力の合計です。
            x_torque,  # ローターの推力によって生成されるトルクです。
            y_torque,  # ローターの推力によって生成されるトルクです。
            z_torque,  # 4つのローターのトルクの合計です。
        )
        """
        forces = np.array(rpm) ** 2 * self._dp.kf
        thrust = np.sum(forces)
        z_torques = np.array(rpm) ** 2 * self._dp.km
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if self._drone_type == DroneType.QUAD_X:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self._dp.l / np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self._dp.l / np.sqrt(2))
        elif self._drone_type in [DroneType.QUAD_PLUS, DroneType.OTHER]:
            x_torque = (forces[1] - forces[3]) * self._dp.l
            y_torque = (-forces[0] + forces[2]) * self._dp.l
        return thrust, x_torque, y_torque, z_torque

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


if __name__ == "__main__":
    '''
    If you want to run this module by itself, try the following.

       $ python -m blt_env.drone

    '''

    # urdf_file = './assets/drone_x_01.urdf'
    # urdf_file = './assets/a.urdf'
    urdf_file = './assets/b.urdf'
    drone_type = DroneType.QUAD_X
    phy_mode = PhysicsType.PYB
    # phy_mode = PhysicsType.DYN

    env = DroneBltEnv(
        urdf_path=urdf_file,
        d_type=drone_type,
        is_gui=True,
        phy_mode=phy_mode,
        num_drones=2,
    )

    env.printout_drone_properties()

    rpms = np.array([14600, 14600, 14600, 14600])

    step_num = 1_000
    for _ in range(step_num):
        ki = env.step(rpms)
        # print(ki)
        time.sleep(env.get_sim_time_step())

    env.close()
