import autograd.numpy as np
from autograd.numpy import sin, cos

def skew(theta):
    theta_hat = np.array([
        [0, -theta[2], theta[1]],
        [theta[2], 0, -theta[0]],
        [-theta[1], theta[0], 0]
    ])
    return theta_hat

def Omega(omega):
    res = np.zeros((4, 4))
    res[1:, 0] = omega
    res[0, 1:] = -omega
    res[1:, 1:] = -skew(omega)
    return res

def get_beta_mean(alpha, beta):
    return alpha / (alpha + beta)

def get_beta_cov(alpha, beta):
    return (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

def quat_to_rot(q):
    n = np.linalg.norm(q)
    q = q / n
    w, x, y, z = q
    return np.array([
            [1 - 2*(y**2 + z**2),   2*(x*y - z*w),      2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x**2 + z**2),    2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),      1 - 2*(x**2 + y**2)]
        ])

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_Exp(phi):
    theta = np.linalg.norm(phi)
    u = phi / theta
    q = np.zeros(4)
    q[0] = cos(theta/2)
    q[1:] = u * sin(theta/2)
    return q

def quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quat_inv(q):
    n = np.linalg.norm(q)
    return quat_conj(q) / n**2

# 5. 四元数表示旋转, 左右四元数矩阵，与另一种转换到旋转矩阵的方式
def quat_L(q):
    qw, x, y, z = q
    qv = np.array([x, y, z])
    qL = np.zeros((4,4))
    qL[0, 0] = qw
    qL[1:, 0] = qv
    qL[0, 1:] = -qv
    qL[1:, 1:] = qw * np.eye(3) + skew(qv)
    return qL

def quat_R(q):
    qw, x, y, z = q
    qv = np.array([x, y, z])
    qR = np.zeros((4,4))
    qR[0, 0] = qw
    qR[1:, 0] = qv
    qR[0, 1:] = -qv
    qR[1:, 1:] = qw * np.eye(3) - skew(qv)
    return qR

def quat_to_rotmat(q):
    q_inv = quat_inv(q)
    R4 = quat_L(q) @ quat_R(q_inv)
    return R4[1:,1:]


def quat_to_euler(q):
    w, x, y, z = q
    cos_pitch_cos_yaw = 1.0 - 2.0 * (y*y + z*z)
    cos_pitch_sin_yaw = 2.0 * (x*y + w*z) 
    sin_pitch = - 2.0 * (x*z - w*y) 
    cos_pitch = 0.0
    sin_roll_cos_pitch = 2.0 * (y*z + w*x)  
    cos_roll_cos_pitch = 1.0 - 2.0 * (x*x + y*y)

    cos_pitch = np.sqrt(cos_pitch_cos_yaw*cos_pitch_cos_yaw + cos_pitch_sin_yaw*cos_pitch_sin_yaw)
    yaw = np.arctan2(cos_pitch_sin_yaw, cos_pitch_cos_yaw)
    if abs(sin_pitch) >= 1:
        pitch = np.sign(sin_pitch) * np.pi / 2
    else:
        pitch = np.arcsin(sin_pitch)
    roll = np.arctan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
    
    euler = np.array([roll, pitch, yaw])

    return euler

def mtx_w_to_euler_dot(euler):
    r, p, y = euler[0], euler[1], euler[2]
    mtx = np.array([
        [1, (sin(p) * sin(r)) / cos(p), (cos(r) * sin(p)) / cos(p)],
        [0, cos(r), -sin(r)],
        [0, sin(r) / cos(p), cos(r) / cos(p)],
    ])
    return mtx

def euler_to_rot(euler):
    r, p, y = euler[0], euler[1], euler[2]
    Ry = np.array([
        [cos(y), -sin(y), 0],
        [sin(y), cos(y), 0],
        [0, 0, 1],
    ])
    Rp = np.array([
        [cos(p), 0, sin(p)],
        [0, 1, 0],
        [-sin(p), 0, cos(p)],
    ])
    Rr = np.array([
        [1, 0, 0],
        [0, cos(r), -sin(r)],
        [0, sin(r), cos(r)],
    ])
    R = Ry @ Rp @ Rr
    return R