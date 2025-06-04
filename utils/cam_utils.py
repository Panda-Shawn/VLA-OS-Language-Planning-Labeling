import numpy as np
from scipy.spatial.transform import Rotation as R


def calculate_2d_position(
    gripper_pos, camera_pos, camera_quat, camera_intrinsics, scalar_first=False
):
    """
    将 3D gripper 位置转换为图像平面上的 2D 坐标。

    参数:
        gripper_pos (np.ndarray): 3D gripper 位置 [x, y, z]，在世界坐标系中。
        camera_pos (np.ndarray): 相机位置 [x, y, z]，在世界坐标系中。
        camera_quat (np.ndarray): 相机方向的四元数 [qx, qy, qz, qw]。
        camera_intrinsics (np.ndarray): 相机内参矩阵 (3x3)，格式如下：
            [[fx,  0, cx],
             [ 0, fy, cy],
             [ 0,  0,  1]]

    返回:
        (u, v): 夹爪在图像平面的像素坐标 (u, v)。
    """
    # Step 1: 将四元数转换为旋转矩阵
    R_camera = R.from_quat(camera_quat, scalar_first=scalar_first).as_matrix()

    # Step 2: 计算相机坐标系中的 gripper 位置
    # 世界坐标系到相机坐标系转换：p_camera = R.T * (p_world - t)
    t_camera = np.array(camera_pos)  # 相机位置 (平移向量)
    gripper_pos = np.array(gripper_pos)  # 夹爪位置 (世界坐标系)
    gripper_camera = R_camera.T @ (gripper_pos - t_camera)

    # Step 3: 投影到图像平面
    # 相机内参矩阵：K @ [x, y, z]^T -> [u * z, v * z, z]
    fx, fy, cx, cy = (
        camera_intrinsics[0, 0],
        camera_intrinsics[1, 1],
        camera_intrinsics[0, 2],
        camera_intrinsics[1, 2],
    )
    x, y, z = gripper_camera  # 相机坐标系中的 3D 点
    u = fx * x / z + cx
    v = fy * y / z + cy
    # import pdb; pdb.set_trace()
    return u, v


def calculate_camera_intrinsics(fovy, resolution):
    """
    根据相机的视场角（fovy）和图像分辨率计算相机内参矩阵。

    参数:
        fovy (float): 垂直视场角（以度为单位）。
        resolution (tuple): 图像分辨率 (width, height)。

    返回:
        np.ndarray: 相机内参矩阵 (3x3)。
    """
    width, height = resolution

    # 计算焦距（像素单位）
    fy = 0.5 * height / np.tan(np.deg2rad(fovy) / 2)
    fx = fy  # 假设像素宽高比为 1:1

    # 光心位于图像中心
    cx = width / 2
    cy = height / 2

    # 构造相机内参矩阵
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return intrinsics


if __name__ == "__main__":
    # 示例数据
    gripper_pos = [0.5, 0.6, 1.2]  # 夹爪位置（世界坐标系）
    camera_pos = [0.6586, 0.0, 1.6103]  # 相机位置（世界坐标系）
    camera_quat = [0.6380, 0.3048, 0.3048, 0.6380]  # 相机方向（四元数）

    # 示例使用
    fovy = 45.0  # 垂直视场角
    resolution = (256, 256)  # 图像分辨率

    camera_intrinsics = calculate_camera_intrinsics(fovy, resolution)
    print("Camera Intrinsics Matrix:")
    print(camera_intrinsics)

    # 计算 2D 坐标
    u, v = calculate_2d_position(
        gripper_pos, camera_pos, camera_quat, camera_intrinsics
    )

    print(f"Gripper's 2D position on the image: u={u}, v={v}")
