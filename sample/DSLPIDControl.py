

class DSLPIDControl(DroneEnvControl): # ドローンの位置および姿勢を制御するためのDSL-PIDコントローラクラス。
    """
    This is a reference from the following ...

        https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/control/DSLPIDControl.py

    """

    def __init__(
            self,
            env: DroneBltEnv,
            pid_coeff: DroneForcePIDCoefficients,
    ):
        super().__init__(env)

        # PID constant parameters
        self._PID = pid_coeff

        # ドローンのPWM（パルス幅変調）から回転数への変換係数と定数、PWMの最小値と最大値を設定
        self._pwm2rpm_scale = 0.2685
        self._pwm2rpm_const = 4070.3
        self._min_pwm = 20000
        self._max_pwm = 65535

        # シミュレーションの時間ステップ、ドローンの特性、重力などを取得
        self._time_step = self._env.get_sim_time_step()
        self._dp = self._env.get_drone_properties()
        self._g = self._dp.g # 重力加速度
        self._mass = self._dp.m # ドローンの質量
        self._kf = self._dp.kf # 推力係数
        self._km = self._dp.km # トルク係数
        self._Mixer = self._dp.Mixer # モーターミキサー
        self._gf = self._dp.gf # 重力

        # Initialized PID control variables PID制御の変数を初期化
        self._last_rpy = np.zeros(3) # 直前の姿勢（ロール、ピッチ、ヨー）
        self._last_pos_e = np.zeros(3) # 直前の位置の誤差
        self._integral_pos_e = np.zeros(3) # 位置の誤差の積分
        self._last_rpy_e = np.zeros(3) # 直前の姿勢の誤差
        self._integral_rpy_e = np.zeros(3) # 姿勢の誤差の積分

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

    def compute_control_from_observation(
            self,
            control_timestep: float,
            obs_state: np.ndarray,
            target_position: np.ndarray,
            target_velocity: np.ndarray = np.zeros(3),
            target_rpy: np.ndarray = np.zeros(3),
            target_rpy_rates: np.ndarray = np.zeros(3),
    ) -> Tuple:
        """ Computes the PID control action (as RPMs) for a single drone.
        Parameters
        ----------
        control_timestep: The time step at which control is computed. 制御が計算されるタイムステップ
        obs_state: (20,)-shaped array of floats containing the current state of the drone. ドローンの現在の状態を含む(20,)-shapedの浮動小数点配列。
        target_position: (3,1)-shaped array of floats containing the desired position. 望ましい位置を含む(3,1)-shapedの浮動小数点配列。
        < The following are optionals. >  
        target_velocity: (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw. 望ましい速度を含む(3,1)-shapedの浮動小数点配列。デフォルトはゼロ。
        target_rpy: (3,1)-shaped array of floats containing the desired velocity. 望ましい姿勢（ロール、ピッチ、ヨー）を含む(3,1)-shapedの浮動小数点配列。デフォルトはゼロ。
        target_rpy_rates: (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates. 望ましいロール、ピッチ、ヨーレートを含む(3,1)-shapedの浮動小数点配列。デフォルトはゼロ。
        """
        return self.compute_control(
            control_timestep=control_timestep,
            current_position=obs_state[0:3],
            current_quaternion=obs_state[3:7],
            current_velocity=obs_state[10:13],
            current_ang_velocity=obs_state[13:16],
            target_position=target_position,
            target_velocity=target_velocity,
            target_rpy=target_rpy,
            target_rpy_rates=target_rpy_rates,
        )

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
        """ Computes the PID control action (as RPMs) for a single drone. 単一のドローンに対するPID制御アクション（RPMとして）を計算します。

        Parameters
        ----------
        control_timestep: The time step at which control is computed.
        current_position: (3,1)-shaped array of floats containing the current position.
        current_quaternion: (4,1)-shaped array of floats containing the current orientation as a quaternion.
        current_velocity: (3,1)-shaped array of floats containing the current velocity.
        current_ang_velocity: (3,1)-shaped array of floats containing the current angular velocity.
        target_position: (3,1)-shaped array of floats containing the desired position.
        < The following are optionals. >
        target_velocity: (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_rpy: (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates: (3,1)-shaped array of floats containing the desired roll, pitch, and yaw rates.

        control_timestep: 制御が計算されるタイムステップ。
        current_position: 現在の位置を含む(3,1)-shapedの浮動小数点配列。
        current_quaternion: 現在の姿勢を含む(4,1)-shapedの浮動小数点配列。
        current_velocity: 現在の速度を含む(3,1)-shapedの浮動小数点配列。
        current_ang_velocity: 現在の角速度を含む(3,1)-shapedの浮動小数点配列。
        target_position: 望ましい位置を含む(3,1)-shapedの浮動小数点配列。
        target_velocity: 望ましい速度を含む(3,1)-shapedの浮動小数点配列。デフォルトはゼロ。
        target_rpy: 望ましい姿勢（ロール、ピッチ、ヨー）を含む(3,1)-shapedの浮動小数点配列。デフォルトはゼロ。
        target_rpy_rates: 望ましいロール、ピッチ、ヨーレートを含む(3,1)-shapedの浮動小数点配列。デフォルトはゼロ。

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray 
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.
        """
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

        self._integral_pos_e = self._integral_pos_e + pos_e * control_timestep # 位置エラーの積分項を更新
        self._integral_pos_e = np.clip(self._integral_pos_e, -2., 2.)
        self._integral_pos_e[2] = np.clip(self._integral_pos_e[2], -0.15, 0.15)

        # PID target thrust # PID制御で求めた目標推力
        target_thrust = np.multiply(self._PID.P_for, pos_e) \
                        + np.multiply(self._PID.I_for, self._integral_pos_e) \
                        + np.multiply(self._PID.D_for, vel_e) \
                        + np.array([0, 0, self._gf])
        scalar_thrust = max(0, np.dot(target_thrust, cur_rotation[:, 2])) # 推力をスカラーに変換
        thrust = (math.sqrt(scalar_thrust / (4 * self._kf)) - self._pwm2rpm_const) / self._pwm2rpm_scale
        target_z_ax = target_thrust / np.linalg.norm(target_thrust)  # 目標Z軸を計算
        target_x_c = np.array([math.cos(target_rpy[2]), math.sin(target_rpy[2]), 0])
        target_y_ax = np.cross(target_z_ax, target_x_c) / np.linalg.norm(np.cross(target_z_ax, target_x_c))
        target_x_ax = np.cross(target_y_ax, target_z_ax)
        target_rotation = (np.vstack([target_x_ax, target_y_ax, target_z_ax])).transpose()

        # Target rotation 
        target_euler = (Rotation.from_matrix(target_rotation)).as_euler('XYZ', degrees=False) # 目標回転
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
                                                                                    target_rotation)
        rot_e = np.array([rot_matrix_e[2, 1], rot_matrix_e[0, 2], rot_matrix_e[1, 0]])
        rpy_rates_e = target_rpy_rates - (cur_rpy - self._last_rpy) / control_timestep # 目標角速度エラーを計算
        self._last_rpy = cur_rpy # 現在のオイラー角を更新
        self._integral_rpy_e = self._integral_rpy_e - rot_e * control_timestep # 角速度エラーの積分項を更新
        self._integral_rpy_e = np.clip(self._integral_rpy_e, -1500., 1500.)
        self._integral_rpy_e[0:2] = np.clip(self._integral_rpy_e[0:2], -1., 1.)
        # PID target torques # PID制御で求めた目標トルク
        target_torques = - np.multiply(self._PID.P_tor, rot_e) \
                         + np.multiply(self._PID.D_tor, rpy_rates_e) \
                         + np.multiply(self._PID.I_tor, self._integral_rpy_e)

        target_torques = np.clip(target_torques, -3200, 3200) # トルクを制約内にクリップ
        pwm = thrust + np.dot(self._Mixer, target_torques) # 推力とトルクからPWMを計算
        pwm = np.clip(pwm, self._min_pwm, self._max_pwm)
        return self._pwm2rpm_scale * pwm + self._pwm2rpm_const # PWMからRPMに変換して返す

    def one_2_3_dim_interface(self, thrust):
        """
        :param thrust:
            Array of floats of length 1, 2, or 4 containing a desired thrust input.
        :return: 4つのモーターそれぞれに適用するPWM値（RPMではない）を計算して返しま
            (4,1)-shaped array of integers containing the PWM (not RPMs) to apply to each of the 4 motors.
        """
        dim = len(np.array(thrust))
        pwm = np.clip(
            (np.sqrt(np.array(thrust) / (self._kf * (4 / dim))) - self._pwm2rpm_const) / self._pwm2rpm_scale,
            self._min_pwm,
            self._max_pwm,
        )
        assert dim in [1, 2, 4], f'in one_2_3_dim_interface()'

        if dim in [1, 4]:
            return np.repeat(pwm, 4 / dim)
        elif dim == 2:
            return np.hstack([pwm, np.flip(pwm)])