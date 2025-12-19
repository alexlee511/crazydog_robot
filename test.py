import mujoco
import mujoco.viewer
import time
import numpy as np
from numpy.linalg import inv, eig
import matplotlib.pyplot as plt



IMU_QUAT_ADR      = 18
IMU_GYRO_ADR      = 22
FRAME_POS_ADR     = 28
FRAME_LINVEL_ADR  = 31

PITCH_REF = 0.0
X_REF = None

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

NUM_MOTOR = 6
# Joint index 對應（方便閱讀）
L_THIGH, L_CALF, L_WHEEL, R_THIGH, R_CALF, R_WHEEL = range(6)


def quat_to_pitch(q):
    qw, qx, qy, qz = q
    # Y 軸 pitch，近似公式（Z-Y-X 或 X-Y-Z 會有點差，但小角沒差太多）
    sinp = 2 * (qw * qy - qz * qx)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    return pitch

def get_state_vector(model, data):
    global PITCH_REF, X_REF
    imu_quat_idx = IMU_QUAT_ADR      # 找到 "imu_quat" 的起始 index
    imu_gyro_idx = IMU_GYRO_ADR      # 找到 "imu_gyro" 的起始 index
    frame_pos_idx = FRAME_POS_ADR     # 找到 "frame_pos" 的起始 index
    frame_linvel_idx = FRAME_LINVEL_ADR  # 找到 "frame_lin_vel" 的起始 index

    imu_quat = data.sensordata[imu_quat_idx:imu_quat_idx+4]
    pitch_abs = quat_to_pitch(imu_quat)

    # --- 1) 在 1s 之後、姿態沒有大晃動時，鎖定一次 PITCH_REF ---
    if (data.time > 1.0) and (PITCH_REF == 0.0) and (abs(pitch_abs) < 0.3):
        PITCH_REF = pitch_abs
        print("Set PITCH_REF =", PITCH_REF)

    imu_gyro = data.sensordata[imu_gyro_idx:imu_gyro_idx+3]
    pitch_rate = imu_gyro[1]   # y 軸為 pitch rate（跟你原本寫的一樣）

    frame_pos = data.sensordata[frame_pos_idx:frame_pos_idx+3]
    frame_lin_vel = data.sensordata[frame_linvel_idx:frame_linvel_idx+3]

    x_abs = frame_pos[0]
    x_dot = frame_lin_vel[0]

    theta = pitch_abs - PITCH_REF

    if (X_REF is None) and (data.time > 1.0) and (abs(theta) < 0.05):
        X_REF = x_abs
        print("Set X_REF =", X_REF)

    if X_REF is None:
        x_pos = 0.0
    else:
        x_pos = x_abs - X_REF


    x = np.array([[x_pos],
                  [x_dot],
                  [theta],
                  [pitch_rate]])
    return x

def solve_DARE(A, B, Q, R, maxiter=150, eps=0.01):
    """
    Solve a discrete time Algebraic Riccati equation (DARE)
    """
    P = Q.copy()
    for i in range(maxiter):
        Pn = A.T @ P @ A - A.T @ P @ B @ inv(R + B.T @ P @ B) @ B.T @ P @ A + Q
        if (np.abs(Pn - P)).max() < eps:
            break
        P = Pn
    return Pn

def dlqr(A, B, Q, R):
    """
    Discrete-time LQR.
    """
    P = solve_DARE(A, B, Q, R)
    K = inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    eigVals, eigVecs = eig(A - B @ K)
    return K, P, eigVals

def build_lqr_for_crazydog(dt):
    g = 9.8

    # ----- 從白板參數來 -----
    Mw_single = 0.28       # 單輪質量 [kg]
    m_body    = 7.08       # 本體質量 [kg]
    l         = 0.37       # 輪軸到本體 COM 的距離 [m]
    r         = 0.07       # 輪半徑 [m]

    # 等效「車體質量」(兩輪 + 轉動慣量折算)
    M_base = 3.0 * Mw_single   # 2*M + 2*(1/2*M) = 3M ≈ 0.84 kg

    # 把 crazydog 當成 cart-pole：
    M = M_base   # cart mass
    m = m_body   # pendulum mass

    # 連續時間狀態空間 (仿照 cart-pole 線性化模型)
    A_c = np.array([
        [0.0, 1.0,          0.0,              0.0],
        [0.0, 0.0,  (m * g) / M,              0.0],
        [0.0, 0.0,          0.0,              1.0],
        [0.0, 0.0, g * (M + m) / (l * M),     0.0]
    ])

    B_c = np.array([[0.0],
                [1.0/M],
                [0.0],
                [1.0/(l*M)]])

    # Euler 離散化
    A = np.eye(4) + dt * A_c
    B = dt * B_c

    # LQR 權重：可以再慢慢調
    Q = np.diag([0.5, 0.1, 150.0, 20.0])   # x, x_dot, theta, theta_dot
    R = np.array([[0.5]])                   # R 大一點，輪子不要太暴力

    K, P, eigVals = dlqr(A, B, Q, R)
    return K


# Load a sample model 
model = mujoco.MjModel.from_xml_path('/home/alexlee/mujoco_course/crazydog_urdf/urdf/scene.xml')
data = mujoco.MjData(model)

target_dof_pos = np.array([1.27, -2.127, 0, 1.27, -2.127, 0])
simulation_dt = 0.005
kps = np.array([25, 25, 0.0, 25, 25, 0.0])   # 兩個輪子 Kp=0
kds = np.array([0.5, 0.5, 0.0, 0.5, 0.5, 0.0])  # 輪子 Kd=0

K_lqr = build_lqr_for_crazydog(simulation_dt)
K_gyro = 20.0

theta_log = []
u_log = []
time_log = []
# Run a simple simulation
with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        step_start = time.time()

        ##controller
        # === 讀取關節狀態 ===
        q  = data.sensordata[0:NUM_MOTOR]          # 0..5: 關節位置
        dq = data.sensordata[NUM_MOTOR:2*NUM_MOTOR]# 6..11: 關節速度

        # === 讀 IMU pitch ===
        imu_quat = data.sensordata[IMU_QUAT_ADR:IMU_QUAT_ADR+4]
        # pitch_abs = quat_to_pitch(imu_quat)


        # # === 讀取 IMU 角速度 ===

        x = get_state_vector(model, data)

        theta = float(x[2, 0])  # 第三個狀態就是 theta
        


        u = -K_lqr @ x    # u 是水平力 F
        # print("k=",K_lqr)
        # print("x=",x)
        # print("u=",u)
        F = float(u[0, 0])
        # F = u

        r= 0.07
        wheel_torque = -F        # 轉成輪子扭矩 τ
        # wheel_torque = np.clip(wheel_torque, -3.0, 3.0)
        print("wheel_torque: ", wheel_torque)
            
        # u = -K_lqr @ x      # (1x4) * (4x1) = (1x1)
        # wheel_torque = -float(u[0, 0])


        # === 統一用 pd_control 算出 6 個關節 torque ===
        target_dof_vel = np.zeros(NUM_MOTOR)
        tau = pd_control(
            target_q  = target_dof_pos,  # 腿追角度、輪子 Kp=0 不看角度
            q         = q,
            kp        = kps,
            target_dq = target_dof_vel,  # 腿目標速度=0，輪子目標速度=wheel_vel_ref
            dq        = dq,
            kd        = kds
        )
        # -------------------------------------------------


        # tau = tau_pd.copy()
        tau[L_WHEEL] = wheel_torque
        tau[R_WHEEL] = wheel_torque

        # tau = pd_control(target_dof_pos, data.sensordata[:NUM_MOTOR], kps, np.zeros(6), data.sensordata[NUM_MOTOR:NUM_MOTOR + NUM_MOTOR], kds)
        data.ctrl[:] = tau
        model.opt.timestep = simulation_dt
        mujoco.mj_step(model, data)
        viewer.sync()

        # --- 紀錄前 20 秒的 theta ---
        if data.time <= 15.0:
            print(f"time: {data.time}, theta: {theta} ")
            time_log.append(data.time)
            theta_log.append(theta)
            u_log.append(F)   

        if data.time >= 15.0:
            break

        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    
# ===== 模擬結束後畫 theta 隨時間變化 =====
fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# 上圖：theta
axs[0].plot(time_log, theta_log)
axs[0].set_title("CrazyDog pitch (theta) over first 15s")
axs[0].set_ylabel("Theta [rad]")
axs[0].grid(True)

# 下圖：u
axs[1].plot(time_log, u_log)
axs[1].set_title("Control input u over first 15s")
axs[1].set_xlabel("Time [s]")
axs[1].set_ylabel("u (F) [N]")
axs[1].grid(True)

plt.show()

