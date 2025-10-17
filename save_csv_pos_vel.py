import mujoco
import mujoco.viewer
import numpy as np
import time
import csv
import os

XML_PATH = "/home/wenconggan/Retarget_Motion_Lafan1/robot_description/x2/x2.xml"
CSV_PATH = "/home/wenconggan/Retarget_Motion_Lafan1/retargeted_motions/x2/walk_all.csv"
PLAY_FPS = 30
OUTPUT_CSV = "/home/wenconggan/Retarget_Motion_Lafan1/retargeted_motions/x2/walk_all_csv.csv"

def load_motion(csv_path):
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            data.append([float(x) for x in row])
    return np.array(data)

motion_data = load_motion(CSV_PATH)
print(f"加载CSV成功，共 {len(motion_data)} 帧, 每帧 {motion_data.shape[1]} 个数值")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

num_root = 7
num_dof = model.nq - num_root
frame_time = 1.0 / PLAY_FPS

print(f"模型总DOF数: {model.nq}，root部分: {num_root}，关节部分: {num_dof}")

def play_and_record(model, data, motion, fps=60, output_csv=None):
    frame_time = 1.0 / fps
    num_frames = len(motion)
    dof_pos_list = []
    dof_vel_list = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i, frame in enumerate(motion):
            root_pos = frame[0:3]
            root_quat = np.array([frame[6], frame[3], frame[4], frame[5]])
            dof_pos = frame[7:7+num_dof]

            data.qpos[:3] = root_pos
            data.qpos[3:7] = root_quat
            data.qpos[7:7+num_dof] = dof_pos

            mujoco.mj_forward(model, data)

            dof_pos_list.append(dof_pos)
            viewer.sync()
            time.sleep(frame_time)
            print(f"帧 {i+1}/{num_frames}")

    # 手动计算速度
    dof_pos_arr = np.array(dof_pos_list)
    dof_vel_arr = np.zeros_like(dof_pos_arr)
    dof_vel_arr[1:] = (dof_pos_arr[1:] - dof_pos_arr[:-1]) / frame_time

    all_dof = np.concatenate([dof_pos_arr, dof_vel_arr], axis=1)

    if output_csv is not None:
        np.savetxt(output_csv, all_dof, delimiter=',', fmt='%.6f')
        print(f"\n✅ 已保存 DOF pos + vel 至: {output_csv}")
        print(f"每帧共 {2*num_dof} 列（前 {num_dof} 为位置，后 {num_dof} 为速度）")

play_and_record(model, data, motion_data, fps=PLAY_FPS, output_csv=OUTPUT_CSV)
