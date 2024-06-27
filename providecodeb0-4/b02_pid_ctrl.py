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
    phy_mode = PhysicsType.PYB_DW # apply_rotor_physics, apply_downwash, pyb_dw
    # phy_mode = PhysicsType.DYN

    init_xyzs = np.array([[0, 0, 1.5]])

    # ドローンの環境を作成
    env = DroneBltEnv( # drone.py
        urdf_path=urdf_file,
        d_type=drone_type,
        is_gui=True,
        phy_mode=phy_mode,
        is_real_time_sim=True,
    )

    # controller ドローンの制御コントローラの初期化
    pid = DroneForcePIDCoefficients(
        P_for=np.array([.4, .4, 1.25]),
        I_for=np.array([.05, .05, .05]),
        D_for=np.array([.2, .2, .5]),
        P_tor=np.array([70000., 70000., 60000.]),
        I_tor=np.array([.0, .0, 500.]),
        D_tor=np.array([20000., 20000., 12000.]),
    )

    ctrl = DSLPIDControl(env, pid_coeff=pid)

    # ドローンの回転数（RPM)設定
    rpms = np.array([14300, 14300, 14300, 14300])

    # Initial target position 初期位置
    pos = np.array([0, 0, 1.0])

    # GUIから目標位置を設定するためのパラメータ
    s_target_x = p.addUserDebugParameter("target_x", -2, 2, pos[0])
    s_target_y = p.addUserDebugParameter("target_y", -2, 2, pos[1])
    s_target_z = p.addUserDebugParameter("target_z", 0, 4, pos[2])

    def get_gui_values():
        tg_x = p.readUserDebugParameter(int(s_target_x))
        tg_y = p.readUserDebugParameter(int(s_target_y))
        tg_z = p.readUserDebugParameter(int(s_target_z))
        return tg_x, tg_y, tg_z

    # # Initialize the logger (optional).
    # d_log = DroneDataLogger(
    #     num_drones=1,
    #     logging_freq=int(env.get_sim_freq()),
    # )

    # シミュレーションステップの実行
    step_num = 2_000
    for i in range(step_num):
        kis = env.step(rpms) # drone.py stepメソッド　ドローンを指定された回転速度で1ステップ進める

        # GUIから目標位置を取得
        tg_x, tg_y, tg_z = get_gui_values()

        # 制御アルゴリズムに基づいて次の回転数（RPM）を計算
        # 与えられた入力に基づいて次の回転数（RPM）を計算し、それをrpmsに格納
        rpms, _, _ = ctrl.compute_control_from_kinematics( # drone_ctrl.pyのメソッド
            control_timestep=env.get_sim_time_step(), # control_timestep: 制御のタイムステップ（時間間隔）　# シミュレーションの時間ステップを取得
            kin_state=kis[0], # kin_state: 現在のドローンの運動学的状態（姿勢や速度など）
            ctrl_target=DroneControlTarget( # ctrl_target: 制御の目標となる値。DroneControlTargetクラスのインスタンスで、目標位置が指定されている
                pos=np.array([tg_x, tg_y, tg_z]), # DroneControlTarget : data_difinition.py 
            ), # DroneControlTargetのposに格納
        )

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
