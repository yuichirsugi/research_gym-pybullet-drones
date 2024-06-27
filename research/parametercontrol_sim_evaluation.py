import numpy as np
import pybullet as p

from util.data_definition import DroneType, PhysicsType
from util.data_definition import DroneForcePIDCoefficients, DroneControlTarget
from blt_env.drone import DroneBltEnv

from control.drone_ctrl import DSLPIDControl

# # Logger class to store drone status (optional).
# from util.data_logger import DroneDataLogger

if __name__ == "__main__":

    # ドローンのURDFファイルと初期設定
    # urdf_file = './assets/drone_x_01.urdf'
    # urdf_file = './assets/a.urdf'
    urdf_file = './assets/b.urdf'
    drone_type = DroneType.QUAD_X # data_definition
    # phy_mode = PhysicsType.PYB_DW # apply_rotor_physics, apply_downwash, pyb_dw
    phy_mode = PhysicsType.DYN

    init_xyzs = np.array([[0, 0, 0]])

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
        I_for=np.array([.05, .05, .05]),
        # D_for=np.array([.2, .2, .5]), # before adjustment
        D_for=np.array([.2, .2, .5]),
        # P_tor=np.array([70000., 70000., 60000.]), # torque gain before adjustment
        P_tor=np.array([75000., 75000., 60000.]), # torque gain
        I_tor=np.array([0., 0., 500.]), # before adjustment
        # I_tor=np.array([1000., 1000., 1000.]),
        D_tor=np.array([20000., 20000., 12000.]), # before adjustment
        # D_tor=np.array([60000., 60000., 36000.]),
    )

    ctrl = DSLPIDControl(env, pid_coeff=pid) # DSLPIDControl : drone_ctrl.py
    # pid_coeff: DroneForcePIDCoefficients, 

    # ドローンの回転数（RPM)設定
    rpms = np.array([14300, 14300, 14300, 14300])
    # rpms = np.array([1000, 1000, 1000, 1000])
    # rpms = np.array([10, 10, 10, 10])

    # Initial target position 初期位置
    pos = np.array([0, 2.0, 1.0])
    # pos = np.array([2.0, 1.0, 1.0])

    # GUIから目標位置を設定するためのパラメータ
    s_target_x = p.addUserDebugParameter("target_x", -2, 2, pos[0])
    s_target_y = p.addUserDebugParameter("target_y", -2, 2, pos[1])
    s_target_z = p.addUserDebugParameter("target_z", 0, 4, pos[2])
    # p.addUserDebugParameter(name=, rangeMin, rangeMax, startValue)

    def get_gui_values():
        tg_x = p.readUserDebugParameter(int(s_target_x))
        tg_y = p.readUserDebugParameter(int(s_target_y))
        tg_z = p.readUserDebugParameter(int(s_target_z))
        return tg_x, tg_y, tg_z

    # tg_x = 2.0
    # tg_y = 2.0
    # tg_z = 2.0

    # # Initialize the logger (optional).
    # d_log = DroneDataLogger(
    #     num_drones=1,
    #     logging_freq=int(env.get_sim_freq()),
    # )

    # current_position = np.array([0.0, 0.0, 0.0])  # ゼロまたは初期位置外で初期化

    # シミュレーションステップの実行
    step_num = 4_000
    log_interval = 100  # ログを取る間隔

    for i in range(step_num):
        kis = env.step(rpms) # drone.py stepメソッド　ドローンを指定された回転速度で1ステップ進める
        # current_position = kis[0].pos
        # GUIから目標位置を取得
        tg_x, tg_y, tg_z = get_gui_values()

        # tg_x = 0
        # tg_y = 0
        # tg_z = 0
        # while current_position != init_xyzs:
        # 制御アルゴリズムに基づいて次の回転数（RPM）を計算
        # 与えられた入力に基づいて次の回転数（RPM）を計算し、それをrpmsに格納
        rpms, _, _ = ctrl.compute_control_from_kinematics( # drone_ctrl.pyのメソッド
            control_timestep=env.get_sim_time_step(), # control_timestep: 制御のタイムステップ（時間間隔）　# シミュレーションの時間ステップを取得
            kin_state=kis[0], # kin_state: 現在のドローンの運動学的状態（姿勢や速度など）# kin_state: DroneKinematicsInfo : 情報の格納場所　kis[0]　ドローンが一体
            ctrl_target=DroneControlTarget( # ctrl_target: 制御の目標となる値。DroneControlTargetクラスのインスタンスで、目標位置が指定されている
                pos=np.array([tg_x, tg_y, tg_z]), # DroneControlTarget : data_difinition.py 
            ), # DroneControlTargetのposに格納
        )

        if i % log_interval == 0: # i が log_interval の倍数のときに条件が真になり、そのときにログが出力
            # i % log_intervalはi を log_interval で割った余り　それが0になる時
            # 500ステップごとにrpyとxyzの値を出力
            print(f"Step {i}:")
            print("Roll, Pitch, Yaw:", kis[0].rpy)
            print("Position (XYZ):", kis[0].pos)
            print("Velocity (XYZ):", kis[0].vel)
        
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
