import os
import argparse
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
import time
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
import joblib
import glob

G1_ROTATION_AXIS = torch.tensor([[
    [0, 1, 0], # l_hip_pitch 
    [1, 0, 0], # l_hip_roll
    [0, 0, 1], # l_hip_yaw
    [0, 1, 0], # l_knee
    [0, 1, 0], # l_ankle_pitch
    [1, 0, 0], # l_ankle_roll
    [0, 1, 0], # r_hip_pitch
    [1, 0, 0], # r_hip_roll
    [0, 0, 1], # r_hip_yaw
    [0, 1, 0], # r_knee
    [0, 1, 0], # r_ankle_pitch
    [1, 0, 0], # r_ankle_roll
    [0, 0, 1], # waist_yaw_joint
    [1, 0, 0], # waist_roll_joint
    [0, 1, 0], # waist_pitch_joint
    [0, 1, 0], # l_shoulder_pitch
    [1, 0, 0], # l_shoulder_roll
    [0, 0, 1], # l_shoulder_yaw
    [0, 1, 0], # l_elbow
    [0, 1, 0], # r_shoulder_pitch
    [1, 0, 0], # r_shoulder_roll
    [0, 0, 1], # r_shoulder_yaw
    [0, 1, 0], # r_elbow
    ]])

X2_ROTATION_AXIS = torch.tensor([[
    [0, 0, 1],  # l_hip_yaw
    [1, 0, 0],  # l_hip_roll
    [0, 1, 0],  # l_hip_pitch
    [0, 1, 0],  # kneel
    [0, 1, 0],  # ankle
    [0, 0, 1],  # r_hip_yaw
    [1, 0, 0],  # r_hip_roll
    [0, 1, 0],  # r_hip_pitch
    [0, 1, 0],  # kneel
    [0, 1, 0],  # ankle
    [0, 1, 0],  # l_shoulder_pitch
    [1, 0, 0],  # l_roll_pitch
    [0, 0, 1],  # l_yaw_pitch
    [0, 1, 0],  # l_elbow
    [0, 1, 0],  # r_shoulder_pitch
    [1, 0, 0],  # r_roll_pitch
    [0, 0, 1],  # r_yaw_pitch
    [0, 1, 0],  # r_elbow
]])


class MotionPlayer:
    def __init__(self, args):
        # init args
        self.args = args
        if self.args.robot_type == 'g1':
            urdf_path = "robot_description/g1/g1_29dof_rev_1_0.urdf"
            self.robot = pin.RobotWrapper.BuildFromURDF('robot_description/g1/g1_29dof_rev_1_0.urdf', 'robot_description/g1', pin.JointModelFreeFlyer())
        elif self.args.robot_type == 'h1_2':
            urdf_path = "robot_description/h1_2/h1_2_wo_hand.urdf"
        elif self.args.robot_type == 'h1':
            urdf_path = "robot_description/h1/h1.urdf"
        elif self.args.robot_type == 'x2':
            urdf_path = "robot_description/x2/x2.urdf"
        else:
            raise ValueError(f"unknowed robot_type: {args.robot_type}")

        # inital gym
        self.gym = gymapi.acquire_gym()
        # create sim environment
        self.sim = self._create_simulation()
        # add plane
        self._add_ground_plane()
        # load urdf
        self.asset = self._load_urdf(urdf_path)
        # create and add robot
        self.env = self._create_env_with_robot()

    def _create_simulation(self):
        """create physics simulation environment"""
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0 / 30.0
        sim_params.gravity = gymapi.Vec3(0.0, 0, -9.81)
        sim_params.up_axis = gymapi.UP_AXIS_Z
        return self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    def _add_ground_plane(self):
        """add plane"""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)  # Z-up plane
        plane_params.distance = 0                   # the distance from plane to original
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        self.gym.add_ground(self.sim, plane_params)

    def _load_urdf(self, urdf_path):
        """load URDF"""
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF not existent: {urdf_path}")
            
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_root = os.path.dirname(urdf_path)
        asset_file = os.path.basename(urdf_path)
        
        return self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _create_env_with_robot(self):
        """create environment with robot"""
        env = self.gym.create_env(self.sim, 
                                 gymapi.Vec3(-2, 0, -2), 
                                 gymapi.Vec3(2, 2, 2), 
                                 1)
        
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 1.0)  # put on the place with 1 meter high
        self.actor = self.gym.create_actor(env, self.asset, pose, "Robot", 0, 0)
        
        return env

    def set_camera(self, viewer):
        """ set the camera"""
        cam_pos = gymapi.Vec3(3, 2, 3)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    def run_viewer(self):
        # create viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if not self.viewer:
            return

        self.set_camera(self.viewer)
        motion_data_list, file_name_list = self.load_data()

        if len(motion_data_list) != 1:
            print("❌ run_viewer only supports a single motion file. Please specify the file using --file_name.")
            return

        motion_data = motion_data_list[0]  # 取第一个文件数据
        file_name = file_name_list[0]

        print(f"Load Motion File: {file_name}, Frames: {motion_data.shape[0]}")

        root_state_tensor = torch.zeros((1, 13), dtype=torch.float32)

        if args.robot_type == 'g1':
            dof_state_tensor = torch.zeros((29, 2), dtype=torch.float32)
        elif args.robot_type == 'x2':
            dof_state_tensor = torch.zeros((18, 2), dtype=torch.float32)
        else:
            raise ValueError(f"unknowed robot_type: {args.robot_type}")

        root_trans_all = []
        pose_aa_all = []
        dof_pos_all = []
        root_rot_all = []
        rot_vec_all = []

        start_frame = 0
        # max_motion_length = 400  # select motion_data
        max_motion_length = motion_data.shape[0] #all motion_data

        def print_progress_bar(current, total, bar_length=40):
            fraction = current / total
            filled_length = int(bar_length * fraction)
            bar = '#' * filled_length + '-' * (bar_length - filled_length)
            print(f"\rRendering Progress: |{bar}| {current}/{total}", end='', flush=True)

        while not self.gym.query_viewer_has_closed(self.viewer):
            for frame_nr in range(start_frame,max_motion_length):
                start_time = time.time()
                configuration = torch.from_numpy(motion_data[frame_nr, :])
                print_progress_bar(frame_nr + 1, max_motion_length)
                root_trans_all.append(configuration[:3])
                root_rot_all.append(configuration[3:7])
                dof_pos_all.append(configuration[7:])
                rotation = R.from_quat(configuration[3:7])
                rotvec = rotation.as_rotvec()
                rotvec = torch.from_numpy(rotvec)
                rot_vec_all.append(rotvec)
                root_state_tensor[0, :7] = configuration[:7]
                dof_state_tensor[:, 0] = configuration[7:]
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_state_tensor))
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(dof_state_tensor))
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                elapsed_time = time.time() - start_time
                sleep_time = max(0, 1.0 / 30.0 - elapsed_time)
                time.sleep(sleep_time)

            print()
            root_trans_all = torch.cat(root_trans_all, dim=0).view(-1, 3).float()
            root_rot_all = torch.cat(root_rot_all, dim=0).view(-1, 4).float()


            if args.robot_type == 'g1':
                dof_pos_all = torch.cat(dof_pos_all, dim=0).view(-1, 29).float()
                dof_pos_all = torch.cat((dof_pos_all[:, :19], dof_pos_all[:, 22:26]), dim=1).float()
                rotation_axis = G1_ROTATION_AXIS
            elif args.robot_type == 'x2':
                dof_pos_all = torch.stack(dof_pos_all).float()
                rotation_axis = X2_ROTATION_AXIS
            else:
                raise ValueError(f"unknowed robot_type: {args.robot_type}")


            rot_vec_all = torch.cat(rot_vec_all, dim=0).view(-1, 3).float()
            N = rot_vec_all.shape[0]
            pose_aa = torch.cat([rot_vec_all[None, :, None], rotation_axis,
                                 torch.zeros((1, N, 3, 3))], dim=2)
            data_name = self.args.robot_type + '_' + self.args.file_name
            data_dump = {}
            data_dump[data_name] = {
                "root_trans_offset": root_trans_all.cpu().detach().numpy(),
                "pose_aa": pose_aa.squeeze().cpu().detach().numpy(),
                "dof": dof_pos_all.detach().cpu().numpy(),
                "root_rot": root_rot_all.cpu().numpy(),
                "fps": 30
            }
            file_base = os.path.splitext(self.args.file_name)[0]
            joblib.dump(data_dump, "pkl_data/" + self.args.robot_type + "/" + file_base + ".pkl")
            print("retargte data save succefully!")
            break

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def load_data(self):
        file_name = self.args.file_name
        robot_type = self.args.robot_type
        base_path = os.path.join('retargeted_motions', robot_type)

        file_path = os.path.join(base_path, file_name)
        matched_files = sorted(glob.glob(file_path))

        if not matched_files:
            raise FileNotFoundError(f"没有找到匹配的 CSV 文件: {file_path}")

        data_list = []
        file_name_list = []

        for csv_file in matched_files:
            data = np.genfromtxt(csv_file, delimiter=',')
            data_list.append(data)
            file_name_list.append(os.path.basename(csv_file))

        return data_list, file_name_list

    # key point visualization
    def clear_lines(self):
        self.gym.clear_lines(self.viewer)

    def draw_sphere(self, pos, radius, color, env_id, pos_id=None):
        sphere_geom_marker = gymutil.WireframeSphereGeometry(radius, 20, 20, None, color=color)
        sphere_pose = gymapi.Transform(gymapi.Vec3(pos[0], pos[1], pos[2]), r=None)
        gymutil.draw_lines(sphere_geom_marker, self.gym, self.viewer, self.env, sphere_pose)

    def draw_line(self, start_point, end_point, color, env_id):
        gymutil.draw_line(start_point, end_point, color, self.gym, self.viewer, self.env)
    
    # get key point by piconicon
    def get_key_point(self, configuration = None):
        self.clear_lines()
        self.robot.framesForwardKinematics(self.Tpose if configuration is None else configuration)
        
        _rot_vec = []
        _root_trans = []
        _root_rot = []
        
        for visual in self.robot.visual_model.geometryObjects:
            frame_name = visual.name[:-2]
            frame_id = self.robot.model.getFrameId(frame_name)
            parent_joint_id = self.robot.model.frames[frame_id].parentJoint
            parent_joint_name = self.robot.model.names[parent_joint_id]
            joint_tf = self.robot.data.oMi[parent_joint_id]
            ref_body_pos = joint_tf.translation 
            ref_body_rot = joint_tf.rotation
            rotation = R.from_matrix(ref_body_rot)
            rot_vec = rotation.as_rotvec()

            if frame_name == 'pelvis':
                _rot_vec = rot_vec
                _root_trans = ref_body_pos
                _root_rot = ref_body_rot
            color_inner = (0.0, 0.0, 0.545)
            color_inner = tuple(color_inner)

            # import ipdb; ipdb.set_trace()
            self.draw_sphere(ref_body_pos, 0.04, color_inner, 0)
            
        return np.array(_root_trans), np.array(_root_rot), np.array(_rot_vec)

    def get_robot_state(self):
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)

        self.num_bodies = self.gym.get_asset_rigid_body_count(self.asset)
        self.num_envs = 1
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)
        self._rigid_body_pos = self._rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 10:13]
        
    def save_data(self):
        pass

    def convert_only(self):
        motion_data_list, file_names = self.load_data()

        for motion_data, fname in zip(motion_data_list, file_names):
            root_trans_all = []
            pose_aa_all = []
            dof_pos_all = []
            root_rot_all = []
            rot_vec_all = []

            max_motion_length = motion_data.shape[0]

            for frame_nr in range(max_motion_length):
                configuration = torch.from_numpy(motion_data[frame_nr, :])
                root_trans_all.append(configuration[:3])
                root_rot_all.append(configuration[3:7])
                dof_pos_all.append(configuration[7:])
                rotation = R.from_quat(configuration[3:7])
                rotvec = torch.from_numpy(rotation.as_rotvec())
                rot_vec_all.append(rotvec)

            root_trans_all = torch.stack(root_trans_all).float()
            root_rot_all = torch.stack(root_rot_all).float()

            if args.robot_type == 'g1':
                dof_pos_all = torch.cat(dof_pos_all, dim=0).view(-1, 29).float()
                dof_pos_all = torch.cat((dof_pos_all[:, :19], dof_pos_all[:, 22:26]), dim=1).float()
                rotation_axis = G1_ROTATION_AXIS
            elif args.robot_type == 'x2':
                dof_pos_all = torch.stack(dof_pos_all).float()
                rotation_axis = X2_ROTATION_AXIS
            else:
                raise ValueError(f"unknowed robot_type: {args.robot_type}")

            rot_vec_all = torch.stack(rot_vec_all).float()
            N = rot_vec_all.shape[0]

            pose_aa = torch.cat([
                rot_vec_all[None, :, None],
                rotation_axis * dof_pos_all[None, :, :, None],
                torch.zeros((1, N, 3, 3))
            ], dim=2)

            file_base = os.path.splitext(fname)[0]
            data_name = self.args.robot_type + '_' + file_base

            data_dump = {
                data_name: {
                    "root_trans_offset": root_trans_all.cpu().numpy(),
                    "pose_aa": pose_aa.squeeze().cpu().numpy(),
                    "dof": dof_pos_all.cpu().numpy(),
                    "root_rot": root_rot_all.cpu().numpy(),
                    "fps": 30
                }
            }
            save_dir = f"pkl_data/{self.args.robot_type}"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, file_base + ".pkl")
            joblib.dump(data_dump, save_path)
            print(f"[✓] saved：{save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, help="File name", default='dance1_subject3.csv')
    # parser.add_argument('--file_name', type=str, help="File name", default='*.csv')
    parser.add_argument('--robot_type', type=str, help="Robot type", default='x2')
    parser.add_argument('--no_viewer', action='store_true', help="Only convert, no rendering")
    args = parser.parse_args()
    loader = MotionPlayer(args)
    if args.no_viewer:
        loader.convert_only()
    else:
        loader.run_viewer()