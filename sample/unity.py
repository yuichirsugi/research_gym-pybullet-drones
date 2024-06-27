import numpy as np

if run_pos_control:
    vel_sp[0] = (_pos_sp[0] - _pos[0]) * pos_p[0]
    vel_sp[1] = (_pos_sp[1] - _pos[1]) * pos_p[1]

if run_alt_control:
    vel_sp[2] = (_pos_sp[2] - _pos[2]) * pos_p[2]

# make sure velocity setpoint is saturated in xy
vel_norm_xy = math.sqrt(vel_sp[0]**2 + vel_sp[1]**2)

if vel_norm_xy > vel_max[0]:
    # note assumes vel_max[0] == vel_max[1]
    vel_sp[0] = vel_sp[0] * vel_max[0] / vel_norm_xy
    vel_sp[1] = vel_sp[1] * vel_max[1] / vel_norm_xy

# make sure velocity setpoint is saturated in z
if vel_sp[2] < -1.0 * params.vel_max_up:
    vel_sp[2] = -1.0 * params.vel_max_up

if vel_sp[2] > params.vel_max_down:
    vel_sp[2] = params.vel_max_down

if not control_mode.flag_control_position_enabled:
    reset_pos_sp = True

if not control_mode.flag_control_altitude_enabled:
    reset_alt_sp = True

if not control_mode.flag_control_velocity_enabled:
    vel_sp_prev[0] = vel[0]
    vel_sp_prev[1] = vel[1]
    vel_sp[0] = 0.0
    vel_sp[1] = 0.0

if not control_mode.flag_control_climb_rate_enabled:
    vel_sp[2] = 0.0

# TODO: remove this is a pathetic leftover, it's here just to make sure that
# _takeoff_jumped flags are reset
if control_mode.flag_control_manual_enabled or not pos_sp_triplet.current.valid \
        or pos_sp_triplet.current.type != position_setpoint_s.SETPOINT_TYPE_TAKEOFF \
        or not control_mode.flag_armed:
    takeoff_jumped = False
    takeoff_thrust_sp = 0.0

limit_acceleration(dt)

vel_sp_prev = vel_sp.copy()

global_vel_sp = {
    'vx': vel_sp[0],
    'vy': vel_sp[1],
    'vz': vel_sp[2]
}

# publish velocity setpoint
if global_vel_sp_pub is not None:
    orb_publish(ORB_ID.vehicle_global_velocity_setpoint, global_vel_sp_pub, global_vel_sp)
else:
    global_vel_sp_pub = orb_advertise(ORB_ID.vehicle_global_velocity_setpoint, global_vel_sp)

if control_mode.flag_control_climb_rate_enabled or control_mode.flag_control_velocity_enabled or \
        control_mode.flag_control_acceleration_enabled:
    # reset integrals if needed
    if control_mode.flag_control_climb_rate_enabled:
        if reset_int_z:
            reset_int_z = False
            thrust_int[2] = 0.0

    else:
        reset_int_z = True

    if control_mode.flag_control_velocity_enabled:
        if reset_int_xy:
            reset_int_xy = False
            thrust_int[0] = 0.0
            thrust_int[1] = 0.0

    else:
        reset_int_xy = True

    # velocity error
    vel_err = vel_sp - vel

    # thrust vector in NED frame
    thrust_sp = None

    if control_mode.flag_control_acceleration_enabled and pos_sp_triplet.current.acceleration_valid:
        thrust_sp = [pos_sp_triplet.current.a_x, pos_sp_triplet.current.a_y, pos_sp_triplet.current.a_z]

    else:
        thrust_sp = vel_err * params.vel_p + vel_err_d * params.vel_d + thrust_int - [0.0, 0.0, params.thr_hover]

    if pos_sp_triplet.current.type == position_setpoint_s.SETPOINT_TYPE_TAKEOFF \
            and not takeoff_jumped and not control_mode.flag_control_manual_enabled:
        # for jumped takeoffs use special thrust setpoint calculated above
        thrust_sp = [0.0, 0.0, -takeoff_thrust_sp]

if not control_mode.flag_control_velocity_enabled and not control_mode.flag_control_acceleration_enabled:
    thrust_sp[0] = 0.0
    thrust_sp[1] = 0.0

# if still or already on ground command zero xy velocity and zero xy thrust_sp in body frame to consider uneven ground
if vehicle_land_detected.ground_contact:
    # thrust setpoint in body frame
    thrust_sp_body = np.dot(R.T, thrust_sp)

    # we don't want to make any correction in body x and y
    thrust_sp_body[0] = 0.0
    thrust_sp_body[1] = 0.0

    # make sure z component of thrust_sp_body is larger than 0 (positive thrust is downward)
    thrust_sp_body[2] = max(thrust_sp[2], 0.0)

    # convert back to local frame (NED)
    thrust_sp = np.dot(R, thrust_sp_body)

    # set velocity setpoint to zero and reset position
    vel_sp[0] = 0.0
    vel_sp[1] = 0.0
    pos_sp[0] = pos[0]
    pos_sp[1] = pos[1]

if not control_mode.flag_control_climb_rate_enabled and not control_mode.flag_control_acceleration_enabled:
    thrust_sp[2] = 0.0

# limit thrust vector and check for saturation
saturation_xy = False
saturation_z = False

# limit min lift
thr_min = params.thr_min

if not control_mode.flag_control_velocity_enabled and thr_min < 0.0:
    # don't allow downside thrust direction in manual attitude mode
    thr_min = 0.0

tilt_max = params.tilt_max_air
thr_max = params.thr_max

# filter vel_z over 1/8sec
vel_z_lp = vel_z_lp * (1.0 - dt * 8.0) + dt * 8.0 * vel[2]

# filter vel_z change over 1/8sec
vel_z_change = (vel[2] - vel_prev[2]) / dt
acc_z_lp = acc_z_lp * (1.0 - dt * 8.0) + dt * 8.0 * vel_z_change

# We can only run the control if we're already in-air, have a takeoff setpoint,
# or if we're in offboard control.
# Otherwise, we should just bail out
got_takeoff_setpoint = (pos_sp_triplet.current.valid and
                        pos_sp_triplet.current.type == position_setpoint_s.SETPOINT_TYPE_TAKEOFF) or \
                       control_mode.flag_control_offboard_enabled

if not control_mode.flag_control_velocity_enabled and not control_mode.flag_control_acceleration_enabled:
    thrust_sp[0] = 0.0
    thrust_sp[1] = 0.0

# if still or already on ground command zero xy velocity and zero xy thrust_sp in body frame to consider uneven ground
if vehicle_land_detected.ground_contact:
    # thrust setpoint in body frame
    thrust_sp_body = np.dot(R.T, thrust_sp)

    # we don't want to make any correction in body x and y
    thrust_sp_body[0] = 0.0
    thrust_sp_body[1] = 0.0

    # make sure z component of thrust_sp_body is larger than 0 (positive thrust is downward)
    thrust_sp_body[2] = max(thrust_sp[2], 0.0)

    # convert back to local frame (NED)
    thrust_sp = np.dot(R, thrust_sp_body)

    # set velocity setpoint to zero and reset position
    vel_sp[0] = 0.0
    vel_sp[1] = 0.0
    pos_sp[0] = pos[0]
    pos_sp[1] = pos[1]

if not control_mode.flag_control_climb_rate_enabled and not control_mode.flag_control_acceleration_enabled:
    thrust_sp[2] = 0.0

# limit thrust vector and check for saturation
saturation_xy = False
saturation_z = False

# limit min lift
thr_min = params.thr_min

if not control_mode.flag_control_velocity_enabled and thr_min < 0.0:
    # don't allow downside thrust direction in manual attitude mode
    thr_min = 0.0

tilt_max = params.tilt_max_air
thr_max = params.thr_max

# filter vel_z over 1/8sec
vel_z_lp = vel_z_lp * (1.0 - dt * 8.0) + dt * 8.0 * vel[2]

# filter vel_z change over 1/8sec
vel_z_change = (vel[2] - vel_prev[2]) / dt
acc_z_lp = acc_z_lp * (1.0 - dt * 8.0) + dt * 8.0 * vel_z_change

# We can only run the control if we're already in-air, have a takeoff setpoint,
# or if we're in offboard control.
# Otherwise, we should just bail out
got_takeoff_setpoint = (pos_sp_triplet.current.valid and
                        pos_sp_triplet.current.type == position_setpoint_s.SETPOINT_TYPE_TAKEOFF) or \
                       control_mode.flag_control_offboard_enabled

import numpy as np

if control_mode.flag_control_velocity_enabled or control_mode.flag_control_acceleration_enabled:

    # limit max tilt
    if thr_min >= 0.0 and tilt_max < np.pi / 2 - 0.05:

        # absolute horizontal thrust
        thrust_sp_xy_len = np.linalg.norm(thrust_sp[:2])

        if thrust_sp_xy_len > 0.01:
            # max horizontal thrust for given vertical thrust
            thrust_xy_max = -thrust_sp[2] * np.tan(tilt_max)

            if thrust_sp_xy_len > thrust_xy_max:
                k = thrust_xy_max / thrust_sp_xy_len
                thrust_sp[:2] *= k
                # Don't freeze x, y integrals if they both want to throttle down
                saturation_xy = (vel_err[0] * vel_sp[0] < 0.0) and (vel_err[1] * vel_sp[1] < 0.0)

if control_mode.flag_control_climb_rate_enabled and not control_mode.flag_control_velocity_enabled:

    # thrust compensation when vertical velocity but not horizontal velocity is controlled
    att_comp = 0.0

    if R[2, 2] > TILT_COS_MAX:
        att_comp = 1.0 / R[2, 2]

    elif R[2, 2] > 0.0:
        att_comp = ((1.0 / TILT_COS_MAX - 1.0) / TILT_COS_MAX) * R[2, 2] + 1.0
        saturation_z = True

    else:
        att_comp = 1.0
        saturation_z = True

    thrust_sp[2] *= att_comp

# Calculate desired total thrust amount in body z direction.
# To compensate for excess thrust during attitude tracking errors we
# project the desired thrust force vector F onto the real vehicle's thrust axis in NED:
# body thrust axis [0, 0, -1]' rotated by R is: R*[0, 0, -1]' = -R_z
R_z = np.array([R[0, 2], R[1, 2], R[2, 2]])
F = np.array(thrust_sp)
thrust_body_z = np.dot(F, -R_z)

# limit max thrust
if abs(thrust_body_z) > thr_max:
    if thrust_sp[2] < 0.0:
        if -thrust_sp[2] > thr_max:
            # thrust Z component is too large, limit it
            thrust_sp[:2] = 0.0
            thrust_sp[2] = -thr_max
            saturation_xy = vel_err[2] < 0.0
            # Don't freeze altitude integral if it wants to throttle down
            saturation_z = vel_err[2] < 0.0

        else:
            # preserve thrust Z component and lower XY, keeping altitude is more important than position
            thrust_xy_max = np.sqrt(thr_max**2 - thrust_sp[2]**2)
            thrust_xy_abs = np.linalg.norm(thrust_sp[:2])
            k = thrust_xy_max / thrust_xy_abs
            thrust_sp[:2] *= k
            # Don't freeze x, y integrals if they both want to throttle down
            saturation_xy = (vel_err[0] * vel_sp[0] < 0.0) and (vel_err[1] * vel_sp[1] < 0.0)

    else:
        # Z component is positive, going down (Z is positive down in NED), simply limit thrust vector
        k = thr_max / abs(thrust_body_z)
        thrust_sp *= k
        saturation_xy = True
        saturation_z = True

    thrust_body_z = thr_max

_att_sp.thrust = max(thrust_body_z, thr_min)

# Update integrals
if control_mode.flag_control_velocity_enabled and not saturation_xy:
    _thrust_int[0] += vel_err[0] * _params.vel_i[0] * dt
    _thrust_int[1] += vel_err[1] * _params.vel_i[1] * dt

if control_mode.flag_control_climb_rate_enabled and not saturation_z:
    _thrust_int[2] += vel_err[2] * _params.vel_i[2] * dt

# Calculate attitude setpoint from thrust vector
if control_mode.flag_control_velocity_enabled or control_mode.flag_control_acceleration_enabled:
    # Desired body_z axis = -normalize(thrust_vector)
    body_z = -thrust_sp / np.linalg.norm(thrust_sp) if np.linalg.norm(thrust_sp) > SIGMA else np.array([0.0, 0.0, 1.0])

    # Vector of desired yaw direction in XY plane, rotated by PI/2
    y_C = np.array([-np.sin(_att_sp.yaw_body), np.cos(_att_sp.yaw_body), 0.0])

    if abs(body_z[2]) > SIGMA:
        # Desired body_x axis, orthogonal to body_z
        body_x = np.cross(y_C, body_z)
        
        # Keep nose to front while inverted upside down
        if body_z[2] < 0.0:
            body_x = -body_x
        
        body_x /= np.linalg.norm(body_x)  # Normalize

    else:
        # Desired thrust is in XY plane, set X downside to construct correct matrix,
        # but yaw component will not be used actually
        body_x = np.array([1.0, 0.0, 0.0])

    # Desired body_y axis
    body_y = np.cross(body_z, body_x)

    # Fill rotation matrix
    _R_setpoint[:3, 0] = body_x
    _R_setpoint[:3, 1] = body_y
    _R_setpoint[:3, 2] = body_z

    # Copy quaternion setpoint to attitude setpoint topic
    q_sp = Quaternion(matrix=_R_setpoint)
    _att_sp.q_d = [q_sp.w, q_sp.x, q_sp.y, q_sp.z]
    _att_sp.q_d_valid = True

    # Calculate euler angles, for logging only, must not be used for control
    euler = euler_from_matrix(_R_setpoint, 'rxyz')
    _att_sp.roll_body = euler[0]
    _att_sp.pitch_body = euler[1]

elif not control_mode.flag_control_manual_enabled:
    # Autonomous altitude control without position control (failsafe landing),
    # force level attitude, don't change yaw
    _R_setpoint = euler_matrix(0.0, 0.0, _att_sp.yaw_body, 'rxyz')

    # Copy quaternion setpoint to attitude setpoint topic
    q_sp = Quaternion(matrix=_R_setpoint)
    _att_sp.q_d = [q_sp.w, q_sp.x, q_sp.y, q_sp.z]
    _att_sp.q_d_valid = True

    _att_sp.roll_body = 0.0
    _att_sp.pitch_body = 0.0

# Save thrust setpoint for logging
_local_pos_sp.acc_x = thrust_sp[0] * ONE_G
_local_pos_sp.acc_y = thrust_sp[1] * ONE_G
_local_pos_sp.acc_z = thrust_sp[2] * ONE_G

_att_sp.timestamp = hrt_absolute_time()