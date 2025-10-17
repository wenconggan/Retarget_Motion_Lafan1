import mujoco
import mujoco.viewer
import numpy as np
import time
import csv
import json
import os

XML_PATH = "/home/wenconggan/Retarget_Motion_Lafan1/robot_description/x2/x2.xml"
CSV_PATH = "/home/wenconggan/Retarget_Motion_Lafan1/retargeted_motions/x2/walk_all.csv"
PLAY_FPS = 30
OUTPUT_TXT = "/home/wenconggan/Retarget_Motion_Lafan1/retargeted_motions/x2/walk_all_motion.txt"

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

def play_and_record(model, data, motion, fps=60):
    frame_time = 1.0 / fps
    num_frames = len(motion)
    dof_pos_list = []

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for i, frame in enumerate(motion):
            root_pos = frame[0:3]
            root_quat = np.array([frame[6], frame[3], frame[4], frame[5]])  # w,x,y,z
            dof_pos = frame[7:7+num_dof]

            data.qpos[:3] = root_pos
            data.qpos[3:7] = root_quat
            data.qpos[7:7+num_dof] = dof_pos

            mujoco.mj_forward(model, data)

            dof_pos_list.append(dof_pos.copy())
            viewer.sync()
            time.sleep(frame_time)
            print(f"帧 {i+1}/{num_frames}", end="\r")

    dof_pos_arr = np.array(dof_pos_list)
    dof_vel_arr = np.zeros_like(dof_pos_arr)
    dof_vel_arr[1:] = (dof_pos_arr[1:] - dof_pos_arr[:-1]) / frame_time

    all_dof = np.concatenate([dof_pos_arr, dof_vel_arr], axis=1)
    print(f"\n录制完成，共 {num_frames} 帧，生成 {2*num_dof} 列 (pos+vel)")
    return all_dof

def save_as_motiontxt(dof_data, output_path, frame_dt):

    dir_path = os.path.dirname(output_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"已创建文件夹：{dir_path}")

    if os.path.exists(output_path):
        print(f"目标文件已存在，将覆盖：{output_path}")
    else:
        print(f"创建新文件：{output_path}")

    frames = dof_data.tolist()
    motion_data = {
        "LoopMode": "Wrap",
        "FrameDuration": frame_dt,
        "EnableCycleOffsetPosition": True,
        "EnableCycleOffsetRotation": True,
        "MotionWeight": 0.5,
        "Frames": frames
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(motion_data, f, indent=2, ensure_ascii=False)

    print(f"已保存 motion.txt，共 {len(frames)} 帧")

dof_data = play_and_record(model, data, motion_data, fps=PLAY_FPS)
save_as_motiontxt(dof_data, OUTPUT_TXT, frame_time)
