import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pinocchio as pin
import torch
import yaml
from pandas.core.arraylike import default_array_ufunc
from tqdm import tqdm

class RetargetingConfig:
    def __init__(self, config_path: str):
        """Load retargeting configuration from YAML file"""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.source_robot = config["source_robot"]
        self.target_robot = config["target_robot"]
        self.height_offset = config.get("height_offset", 0.0)
        self.joint_mapping = config["joint_pairs"]


class MotionRetargeting:
    def __init__(
            self, config: RetargetingConfig, source_urdf: str, target_urdf: str
    ):
        self.config = config
        self.source_model, self.source_data = self.load_robot_model(source_urdf)
        self.target_model, self.target_data = self.load_robot_model(target_urdf)

        # Get joint IDs for mapping
        self.source_joint_ids = [
            self.source_model.getJointId(name)
            for name in self.config.joint_mapping.keys()
        ]
        self.target_joint_ids = [
            self.target_model.getJointId(name)
            for name in self.config.joint_mapping.values()
        ]

    @staticmethod
    def load_robot_model(urdf_path: str) -> Tuple[pin.Model, pin.Data]:
        """Load robot model from URDF"""
        robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path,
            os.path.dirname(urdf_path),
            pin.JointModelFreeFlyer(),
        )
        return robot.model, robot.data

    @staticmethod
    def load_motion_data(file_name):
        motion = np.loadtxt(file_name, delimiter=",")
        return torch.from_numpy(motion).float()

    @staticmethod
    def save_motion_data(optimized_angles, output_file, joint_names=None):
        """Save optimized angles as CSV without header"""
        if joint_names is not None:
            header = ",".join(joint_names)
            # Save raw data
            np.savetxt(
                output_file,
                optimized_angles,
                fmt="%.6f",  # 6 decimal precision
                delimiter=",",
                comments="",  # No comments/header
                header=header,
            )
        else:
            np.savetxt(
                output_file,
                optimized_angles,
                fmt="%.6f",  # 6 decimal precision
                delimiter=",",
                comments="",  # No comments/header
            )

    @staticmethod
    def clamp_to_joint_limits(model, q):
        """
        Clamp configuration to joint limits
        Args:
            model: Pinocchio model
            q: Configuration to clamp
        Returns:
            Clamped configuration
        """
        # Get joint limits (excluding freeflyer)
        upper_limits = model.upperPositionLimit[7:]
        lower_limits = model.lowerPositionLimit[7:]

        # Keep freeflyer as is
        q_clamped = q.copy()

        # Clamp joint angles to limits
        q_clamped[7:] = np.clip(q[7:], lower_limits, upper_limits)

        return q_clamped

    def retarget_motion(
            self,
            motion_data: torch.Tensor,
            max_iterations: int = 100,
            damping: float = 1e-12,
            eps: float = 1e-6,
            dt: float = 1e-1,
    ) -> np.ndarray:
        """Retarget motion from source to target robot (with root position integrated)"""
        num_frames = len(motion_data)
        optimized_angles = np.zeros((num_frames, self.target_model.nq))

        q_prev = pin.neutral(self.target_model)
        q_prev[:7] = motion_data[0][:7].numpy()
        q_prev[2] -= self.config.height_offset
        q_prev[7:] = np.zeros(self.target_model.nq - 7)

        with tqdm(
                range(num_frames),
                desc="Retargeting frames",
                position=1,
                leave=False,
        ) as pbar:
            for frame in pbar:
                source_q = motion_data[frame].numpy()
                pin.forwardKinematics(self.source_model, self.source_data, source_q)
                pin.updateFramePlacements(self.source_model, self.source_data)
                source_root_pose = self.source_data.oMi[1]  # 假设根关节ID为1
                q = q_prev.copy()
                q[:7] = source_q[:7]
                q[2] -= self.config.height_offset

                target_relative_poses = [
                    source_root_pose.inverse() * self.source_data.oMi[joint_id]
                    for joint_id in self.source_joint_ids
                ]

                for _ in range(max_iterations):
                    pin.forwardKinematics(self.target_model, self.target_data, q)
                    pin.updateFramePlacements(self.target_model, self.target_data)
                    target_root_pose = self.target_data.oMi[1]
                    current_relative_poses = [
                        target_root_pose.inverse() * self.target_data.oMi[joint_id]
                        for joint_id in self.target_joint_ids
                    ]

                    errors = []
                    for target_rel, current_rel in zip(target_relative_poses, current_relative_poses):
                        err_se3 = pin.log6(current_rel.inverse() * target_rel)

                        errors.append(err_se3)
                    errors_flat = np.concatenate(errors)

                    if np.linalg.norm(errors_flat) < eps:
                        break

                    pin.computeJointJacobians(self.target_model, self.target_data, q)

                    J_list = []
                    for joint_id in self.target_joint_ids:
                        J_joint = pin.getJointJacobian(
                            self.target_model, self.target_data, joint_id,
                            pin.ReferenceFrame.LOCAL
                        )
                        J_joint_non_root = J_joint[:, 6:]
                        J_list.append(J_joint_non_root)
                    J = np.vstack(J_list)

                    JJt = J @ J.T
                    lambda2 = damping * np.eye(JJt.shape[0])
                    dq_non_root = J.T @ np.linalg.solve(JJt + lambda2, errors_flat)
                    dq = np.zeros(self.target_model.nv)  # 总自由度速度
                    dq[6:] = dq_non_root  # 非根关节速度

                    q = pin.integrate(self.target_model, q, dt * dq)
                    q[7:] = np.clip(
                        q[7:],
                        self.target_model.lowerPositionLimit[7:],
                        self.target_model.upperPositionLimit[7:]
                    )
                    q[:7] = source_q[:7]
                    q[2] -= self.config.height_offset

                optimized_angles[frame] = q
                q_prev = q.copy()
                pbar.set_postfix(frame=f"{frame}/{num_frames}")

        return optimized_angles


def main():
    parser = argparse.ArgumentParser(
        description="Motion retargeting between robots"
    )
    parser.add_argument("files", nargs="+", help="Motion files to process")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to joint mapping YAML config",
    )
    parser.add_argument(
        "--source-urdf",
        type=str,
        required=True,
        help="Path to source robot URDF",
    )
    parser.add_argument(
        "--target-urdf",
        type=str,
        required=True,
        help="Path to target robot URDF",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="retargeted_motions",
        help="Output directory for retargeted motions",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = RetargetingConfig(args.config)
    retargeting = MotionRetargeting(config, args.source_urdf, args.target_urdf)

    files = []
    for pattern in args.files:
        files.extend(glob.glob(pattern))

    with tqdm(files, desc="Processing files", position=0) as pbar:
        for file_path in pbar:
            try:
                motion_data = retargeting.load_motion_data(file_path)
                retargeted_motion = retargeting.retarget_motion(motion_data)

                output_file = os.path.join(
                    args.output_dir,
                    f"{os.path.basename(file_path)}",
                )
                retargeting.save_motion_data(retargeted_motion, output_file)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    main()

