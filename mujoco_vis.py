import mujoco
import mujoco.viewer
import numpy as np
import time
import csv
import os

# XML_PATH = "/home/wenconggan/Retarget_Motion_Lafan1/robot_description/x3b/x3.xml"
# CSV_PATH = "/home/wenconggan/Retarget_Motion_Lafan1/retargeted_motions/x3b/jumps1_subject1.csv"

XML_PATH = "/home/wenconggan/Retarget_Motion_Lafan1/robot_description/x2/x2.xml"
CSV_PATH = "/retargeted_motions/x2/walk_all.csv"

PLAY_FPS = 30

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
print(f"模型总DOF数: {model.nq}，root部分: {num_root}，关节部分: {num_dof}")
def play_motion(model, data, motion, fps=60):
    frame_time = 1.0 / fps
    viewer = mujoco.viewer.launch_passive(model, data)
    try:
        for i, frame in enumerate(motion):
            root_pos = frame[0:3]
            root_quat = np.array([frame[6], frame[3], frame[4], frame[5]])
            dof_pos = frame[7:7+num_dof]

            data.qpos[:3] = root_pos
            data.qpos[3:7] = root_quat
            data.qpos[7:7+num_dof] = dof_pos

            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(frame_time)
            print(i,len(motion_data))

    finally:
        viewer.close()


play_motion(model, data, motion_data, fps=PLAY_FPS)
