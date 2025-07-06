import os
import argparse
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
import time

class MotionPlayer:
    def __init__(self, args):
        # init args
        self.args = args
        if self.args.robot_type == 'g1':
            urdf_path = "robot_description/g1/g1_29dof_rev_1_0.urdf"
        elif self.args.robot_type == 'h1_2':
            urdf_path = "robot_description/h1_2/h1_2_wo_hand.urdf"
        elif self.args.robot_type == 'h1':
            urdf_path = "robot_description/h1/h1.urdf"
        elif self.args.robot_type == 'x2':
            urdf_path = "robot_description/x2/x2.urdf"
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
        """run visualize"""
        # create viewer
        viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if not viewer:
            return
            
        self.set_camera(viewer)
        motion_data = self.load_data()
        
        root_state_tensor = torch.zeros((1, 13), dtype=torch.float32)
        dof_state_tensor = torch.zeros((18, 2), dtype=torch.float32)
        
        # main loop
        while not self.gym.query_viewer_has_closed(viewer):
            for frame_nr in range(motion_data.shape[0]):
                start_time = time.time()
                configuration = torch.from_numpy(motion_data[frame_nr, :])
                root_state_tensor[0, :7] = configuration[:7]
                dof_state_tensor[:,0] = configuration[7:]
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_state_tensor))
                self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(dof_state_tensor))
                
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(viewer, self.sim, True)
                elapsed_time = time.time() - start_time
                sleep_time = max(0, 1.0 / 30.0 - elapsed_time)
                time.sleep(sleep_time)
            
        self.gym.destroy_viewer(viewer)
        self.gym.destroy_sim(self.sim)

    def load_data(self):
        file_name = self.args.file_name
        robot_type = self.args.robot_type
        csv_files = 'retargeted_motions/'+robot_type + '/' + file_name
        data = np.genfromtxt(csv_files, delimiter=',')
        
        return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, help="File name", default='dance1_subject2.csv')
    parser.add_argument('--robot_type', type=str, help="Robot type", default='x2')
    args = parser.parse_args()
    loader = MotionPlayer(args)
    loader.run_viewer()