from typing import List, Optional
from dataclasses import dataclass, field
from enum import IntEnum, Enum

import numpy as np

"""
# @dataclass(init=False, frozen=True)
@dataclass(frozen=True)
class FooBarConstants(object):
    bar = 'b_val'
    foo = 'f_val'

    # def __post_init__(self):
    #     self.bar = 'bar_value'  # FrozenInstanceError
    #     object.__setattr__(self, 'foo', 'foo_value') # not get error

"""


@dataclass(init=False, frozen=True)
class SharedConstants(object): # ドローンシミュレーションにおいて共通的に使用される定数を定義
    AGGR_PHY_STEPS: int = 5  # 例えば物理ステップ、成功モデルのデフォルト名、出力ディレクトリのデフォルトパス。デフォルトのドローンファイルのパス
    # for default setting
    SUCCESS_MODEL_FILE_NAME: str = 'success_model.zip'
    DEFAULT_OUTPUT_DIR_PATH: str = './result'
    # DEFAULT_DRONE_FILE_PATH: str = './assets/drone_x_01.urdf'
    # DEFAULT_DRONE_FILE_PATH: str = './assets/a.urdf'
    DEFAULT_DRONE_FILE_PATH: str = './assets/b.urdf'
    DEFAULT_DRONE_TYPE_NAME: str = 'x'


class DroneType(IntEnum): # ドローンのタイプを表す列挙型。QUAD_PLUSとQUAD_Xが含まれる
    OTHER = 0
    QUAD_PLUS = 1
    QUAD_X = 2


class PhysicsType(Enum): # PhysicsType: ドローンの物理実装の種類を表す列挙型。
    """Physics implementations enumeration class."""
    PYB = "pyb"  # Base PyBullet physics update 基本のPyBullet物理エンジン
    DYN = "dyn"  # Update with an explicit model of the dynamics 明示的なダイナミクスモデルを使用したアップデート
    PYB_GND = "pyb_gnd"  # PyBullet physics update with ground effect 地面効果付きのPyBullet物理エンジンアップデート
    PYB_DRAG = "pyb_drag"  # PyBullet physics update with drag ドラッグを考慮したPyBullet物理エンジンアップデート
    PYB_DW = "pyb_dw"  # PyBullet physics update with downwash ダウンウォッシュを考慮したPyBullet物理エンジンアップデート
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw"  # PyBullet physics update with ground effect, drag, and downwash
    # 地面効果、ドラッグ、ダウンウォッシュを考慮したPyBullet物理エンジンアップデート


class ActionType(Enum): # ActionType: ドローンの制御アクションの種類を表す列挙型。
    """Action type enumeration class."""
    RPM = "rpm"  # RPMS 回転数（RPM）
    FORCE = "for"  # Desired thrust and torques (force) 望ましい推力およびトルク（力）
    PID = "pid"  # PID control PID制御 
    VEL = "vel"  # Velocity input (using PID control) 速度入力（PID制御を使用）
    TUN = "tun"  # Tune the coefficients of a PID controller PIDコントローラーの係数を調整
    ONE_D_RPM = "one_d_rpm"  # 1D (identical input to all motors) with RPMs 1次元（全てのモーターに同じ入力）でのRPM
    ONE_D_FORCE = "one_d_for"  # 1D (identical input to all motors) with desired thrust and torques 1次元（全てのモーターに同じ入力）での望ましい推力およびトルク
    ONE_D_PID = "one_d_pid"  # 1D (identical input to all motors) with PID control 1次元（全てのモーターに同じ入力）でのPID制御


class RlAlgorithmType(Enum): # RlAlgorithmType: 強化学習アルゴリズムの種類を表す列挙型。
    """Reinforcement Learning type enumeration class."""
    A2C = 'a2c'
    PPO = 'ppo'
    SAC = 'sac'
    TD3 = 'td3'
    DDPG = 'ddpg'


class ObservationType(Enum): # ObservationType: ドローンの観測タイプを表す列挙型。
    """Observation type enumeration class."""
    KIN = "kin"  # Kinematics information (pose, linear and angular velocities)
    # キネマティクス情報（位置、線形および角速度）
    RGB = "rgb"  # RGB camera capture in each drone's POV
    # 各ドローンの視点でのRGBカメラキャプチャ


@dataclass(frozen=True)
class DroneForcePIDCoefficients(object): # ドローンの力とトルクに関するPID係数を格納するためのクラス
    P_pos_for: np.ndarray = None  # force 推力
    I_pos_for: np.ndarray = None
    D_pos_for: np.ndarray = None
    # P_for: np.ndarray = None  # force 推力
    # I_for: np.ndarray = None
    # D_for: np.ndarray = None
    P_tor: np.ndarray = None  # torque トルク
    I_tor: np.ndarray = None
    D_tor: np.ndarray = None # None 後で何かしら新しい値が代入される
    P_vel_for: np.ndarray = None  # force 推力 なんかあったらここ消す　pos消す
    I_vel_for: np.ndarray = None
    D_vel_for: np.ndarray = None


@dataclass
class DroneKinematicsInfo(object): # ドローンの運動学情報（位置、姿勢、速度、角速度など）を格納するためのクラス
    pos: np.ndarray = np.zeros(3)  # position　位置
    quat: np.ndarray = np.zeros(4)  # quaternion　クウォータニオン
    rpy: np.ndarray = np.zeros(3)  # roll, pitch and yaw
    vel: np.ndarray = np.zeros(3)  # linear velocity　線形速度
    ang_vel: np.ndarray = np.zeros(3)  # angular velocity　角速度


@dataclass
class DroneControlTarget(object): # ドローンの制御対象となる目標情報（位置、速度、姿勢、角速度など）を格納するためのクラス
    pos: np.ndarray = np.zeros(3)  # position
    vel: np.ndarray = np.zeros(3)  # linear velocity　線形速度
    rpy: np.ndarray = np.zeros(3)  # roll, pitch and yaw
    rpy_rates: np.ndarray = np.zeros(3)  # roll, pitch, and yaw rates　レート
    force: np.ndarray = np.zeros(3)


@dataclass
class DroneProperties(object): # ドローンの物理的な特性やパラメータを格納するためのクラス
    """
    The drone parameters.

    kf : It is the proportionality constant for thrust, and thrust is proportional to the square of rotation speed.
    km : It is the proportionality constant for torque, and torque is proportional to the square of rotation speed.

    """
    type: int = 1  # The drone type 0:OTHER 1:QUAD_PLUS 2:QUAD_X
    g: float = 9.8  # gravity acceleration　重力加速度
    m: Optional[float] = None  # Mass of the drone.　ドローンの質量
    l: Optional[float] = None  # Length of the arm of the drone's rotor mount.　ドローンのローターマウントのアームの長さ
    thrust2weight_ratio: Optional[float] = None
    ixx: float = 0
    iyy: float = 0
    izz: float = 0
    J: np.ndarray = np.array([])
    J_inv: np.ndarray = np.array([])
    kf: Optional[float] = None  # The proportionality constant for thrust.　推力の比例定数
    km: Optional[float] = None  # The proportionality constant for torque.　トルクの比例定数
    collision_h: Optional[float] = None
    collision_r: Optional[float] = None
    collision_shape_offsets: List[float] = field(default_factory=list)
    collision_z_offset: float = None
    max_speed_kmh: Optional[float] = None
    gnd_eff_coeff: Optional[float] = None
    prop_radius: Optional[float] = None
    drag_coeff_xy: float = 0
    drag_coeff_z: float = 0
    drag_coeff: np.ndarray = None
    dw_coeff_1: Optional[float] = None
    dw_coeff_2: Optional[float] = None
    dw_coeff_3: Optional[float] = None
    # compute after determining the drone type
    gf: float = 0  # gravity force 重力による力
    hover_rpm: float = 0 # ホバリング時の回転数
    max_rpm: float = 0 # 最大回転数
    max_thrust: float = 0 # 最大推力
    max_xy_torque = 0 # XY平面での最大トルク
    max_z_torque = 0 # Z軸周りでの最大トルク
    grand_eff_h_clip = 0  # The threshold height for ground effects.　地面効果のー高さ
    A: np.ndarray = np.array([])
    inv_A: np.ndarray = np.array([])
    B_coeff: np.ndarray = np.array([])
    Mixer: np.ndarray = np.ndarray([])  # use for PID control PID制御に使用

    def __post_init__(self):
        self.J = np.diag([self.ixx, self.iyy, self.izz]) #　慣性行列を初期化
        self.J_inv = np.linalg.inv(self.J) # 慣性行列の逆行列を計算
        self.collision_z_offset = self.collision_shape_offsets[2] # 衝突形状のzオフセットを設定
        self.drag_coeff = np.array([self.drag_coeff_xy, self.drag_coeff_xy, self.drag_coeff_z]) # ドラッグ係数を設定
        self.gf = self.g * self.m # 重力による力を計算
        self.hover_rpm = np.sqrt(self.gf / (4 * self.kf)) # ホバリング時の回転数を計算
        self.max_rpm = np.sqrt((self.thrust2weight_ratio * self.gf) / (4 * self.kf)) # 最大回転数を計算
        self.max_thrust = (4 * self.kf * self.max_rpm ** 2) # 最大推力を計算
        if self.type == 2:  # QUAD_X
            self.max_xy_torque = (2 * self.l * self.kf * self.max_rpm ** 2) / np.sqrt(2) # XY平面での最大トルクの計算
            self.A = np.array([[1, 1, 1, 1], [1 / np.sqrt(2), 1 / np.sqrt(2), -1 / np.sqrt(2), -1 / np.sqrt(2)],
                               [-1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2), -1 / np.sqrt(2)], [-1, 1, -1, 1]]) # モーターの制御行列を設定
            self.Mixer = np.array([[.5, -.5, -1], [.5, .5, 1], [-.5, .5, -1], [-.5, -.5, 1]]) # モーターの混合行列を設定
        elif self.type in [0, 1]:  # QUAD_PLUS, OTHER
            self.max_xy_torque = (self.l * self.kf * self.max_rpm ** 2) 
            self.A = np.array([[1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1]])
            self.Mixer = np.array([[0, -1, -1], [+1, 0, 1], [0, 1, -1], [-1, 0, 1]])
        self.max_z_torque = 2 * self.km * self.max_rpm ** 2 #  Z軸周りでの最大トルク
        self.grand_eff_h_clip = 0.25 * self.prop_radius * np.sqrt(
            (15 * self.max_rpm ** 2 * self.kf * self.gnd_eff_coeff) / self.max_thrust) # 地面効果の閾値高さを計算
        self.inv_A = np.linalg.inv(self.A) # 制御行列 A の逆行列を計算
        self.B_coeff = np.array([1 / self.kf, 1 / (self.kf * self.l), 1 / (self.kf * self.l), 1 / self.km]) # 制御入力行列の係数を設定
