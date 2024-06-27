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
        # P_for=np.array([.2, .2, 1.25]),
        P_pos_for=np.array([0.2, 0.2, 1.25]),
        # I_for=np.array([.05, .05, .05]), # before adjustment
        # I_for=np.array([.05, .05, .05]),
        I_pos_for=np.array([0, 0, 0]),
        # D_for=np.array([.2, .2, .5]), # before adjustment
        D_pos_for=np.array([.2, .2, .5]),
        # P_tor=np.array([70000., 70000., 60000.]), # torque gain before adjustment
        P_tor=np.array([75000., 75000., 60000.]), # torque gain
        I_tor=np.array([0., 0., 500.]), # before adjustment
        # I_tor=np.array([1000., 1000., 1000.]),
        # I_tor=np.array([250., 250., 500.]),
        D_tor=np.array([20000., 20000., 12000.]), # before adjustment
        # D_tor=np.array([60000., 60000., 36000.]),
        P_vel_for=np.array([0.2, 0.2, 1.25]), # torque gain
        I_vel_for=np.array([0, 0, 0]), 
        D_vel_for=np.array([.2, .2, .5]), # before adjustment
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

    target_vel = np.array([-0.0000001, 0, 0])  # 初期位置とその次の位置を設定

    # target_velocity = [
    #     [0.5, 0.5, 0.5], 
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

    prev_velocity = np.zeros(3)

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
                ),
            )

        else:
            rpms = ctrl.coating_motion_control_from_kinematics( # drone_ctrl.pyのメソッド
                control_timestep=env.get_sim_time_step(), # control_timestep: 制御のタイムステップ（時間間隔）　# シミュレーションの時間ステップを取得
                kin_state=kis, # kin_state: 現在のドローンの運動学的状態（姿勢や速度など）# kin_state: DroneKinematicsInfo : 情報の格納場所　kis[0]　ドローンが一体
                ctrl_target=DroneControlTarget( # ctrl_target: 制御の目標となる値。DroneControlTargetクラスのインスタンスで、目標位置が指定されている
                    vel=target_vel,
                ),
                prev_velocity=prev_velocity, # DroneControlTargetのposに格納
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

        if pos_difference[current_target_index] < pos_tolerance and not flag_condition0:
            print(f"目標位置 {current_target_index} に到達しました。")
            print(f"目標姿勢 {current_target_index} に到達しました。")

            # 次の目標位置のインデックスを更新
            current_target_index += 1

            current_target_pos = target_positions[current_target_index]

            if current_target_index < len(target_positions):
                current_target_pos = target_positions[current_target_index]
                current_target_rpy = target_rpys[current_target_index]

            # フラグをセット
            flag_condition0 = True

        if pos_difference[current_target_index] < pos_tolerance and not flag_condition1:

            print(f"目標位置 {current_target_index} に到達しました。")
            print(f"目標姿勢 {current_target_index} に到達しました。")

            # フラグをセット
            flag_condition1 = True

        
        if flag_condition0 and not flag_condition2:
            print(f"ドローンは {current_target_index} 番目の位置で接触しました。5秒間待機します。")

            # 5秒間待機
            time.sleep(5)

            # 姿勢を維持したまま速度制御を加える
            rpms = ctrl.coating_motion_control_from_kinematics(
                control_timestep=env.get_sim_time_step(),
                kin_state=kis,
                ctrl_target=DroneControlTarget(
                    vel=target_vel,
                ),
                prev_velocity=prev_velocity,
            )

            # フラグをセット
            flag_condition2 = True

        if flag_condition2 and not flag_condition3:
            print(f"ドローンは {current_target_index} 番目の位置から動きます。")

            # 姿勢を平行に戻す
            rpms = ctrl.compute_control_from_kinematics(
                control_timestep=env.get_sim_time_step(),
                kin_state=kis,
                ctrl_target=DroneControlTarget(
                    pos=current_target_pos,
                ),
            )

            # フラグをセット
            flag_condition3 = True

        if flag_condition3 and not flag_condition4:
            print(f"ドローンは {current_target_index} 番目の位置に戻りました。")

            # フラグをセット
            flag_condition4 = True

            # 次の目標位置のインデックスを更新
            current_target_index += 1

            if current_target_index < len(target_positions):
                current_target_pos = target_positions[current_target_index]
                current_target_rpy = target_rpys[current_target_index]

        # 前回の速度へ更新
        prev_velocity = kis.vel

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