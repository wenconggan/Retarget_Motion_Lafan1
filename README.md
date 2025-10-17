#  Retarget_Motion_Lafan1
### Usage


Env Install:

```bash
conda create -n retarget_lafan_env python=3.9
pip install -r requirements.txt
pip install pyyaml
conda install -c conda-forge pinocchio
```


1. Extracting joint names from URDF files
```bash
# Default behavior with built-in URDFs
python extract_joint_names.py
```
```bash
# Custom URDF with verbose output
python extract_joint_names.py -u /path/xxx/xxx.urdf -v
python extract_joint_names.py -u /home/wenconggan/motion_retarget/robot_description/g1/g1_29dof_rev_1_0.urdf -v

```
```bas
# Multiple URDFs without joint limits
python extract_joint_names.py -u robot1.urdf robot2.urdf --no-limits
```


2. Retargeting motion files.

```bash
python retarget_motion.py \
    --config robot_configs/x2.yaml \
    --source-urdf robot_description/g1/g1_29dof_rev_1_0.urdf \
    --target-urdf robot_description/x2/x2.urdf \
    --output-dir retargeted_motions/x2 \
    /home/wenconggan/桌面/Retarget_Motion_Lafan1/retargeted_motions/g1/dance1_subject2.csv

python retarget_motion.py     --config robot_configs/x2.yaml  --target-urdf robot_description/x2/x2.urdf     --output-dir retargeted_motions/x2     /home/wenconggan/Retarget_Motion_Lafan1/g1/dance1_subject1.csv
```


convert all lafan motion 
```
python retarget_motion.py   --config robot_configs/x2.yaml  --target-urdf robot_description/x2/x2.urdf     --output-dir retargeted_motions/x2     /home/xxx/g1/*.csv
```

3. Visualizing retargeted motion files.

rerun:

```bash
# Basic usage
python rerun_visualize.py -m motions/dance1.csv -r g1
python rerun_visualize.py -m motions/dance1.csv -r stompypro
python rerun_visualize.py -m /home/wenconggan/motion_retarget/retargeted_motions/dance1_subject2.csv -r stompypro

```

```bash
# Custom visualization settings
python rerun_visualize.py \
    --motion-file /home/wenconggan/桌面/LAFAN1_Visualize/motion_retarget/retargeted_motions/dance1_subject2.csv \
    --robot-type x2 \
    --frame-delay 0.05 \
    --window-title "X2 Dance" \
    --coordinate-frame LEFT_HAND_Z_UP
```
Mujoco:

```bash
python mujoco_vis.py 
 ```
