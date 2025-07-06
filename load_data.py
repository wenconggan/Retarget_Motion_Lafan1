import pickle
import joblib

# 打开 .pkl 文件进行读取
with open('pkl_data/g1/test.pkl', 'rb') as file:
    data = joblib.load(file)
    
# 打印文件中加载的数据（具体格式取决于文件内容）

# print(data['0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass']['root_trans_offset'].shape)
# print(data['0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass']['pose_aa'].shape)
# print(data['0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass']['dof'].shape)
# print(data['0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass']['root_rot'].shape)
# print(data['0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass']['smpl_joints'].shape)
# print(data.keys())
# print(data['0-TotalCapture_s1_walking1_poses']['root_trans_offset'].shape)
# print(data['0-TotalCapture_s1_walking1_poses']['pose_aa'].shape)
# print(data['0-TotalCapture_s1_walking1_poses']['dof'].shape)
# print(data['0-TotalCapture_s1_walking1_poses']['root_rot'].shape)

print(data['g1_dance1_subject2']['root_trans_offset'].shape)
print(data['g1_dance1_subject2']['pose_aa'].shape)
print(data['g1_dance1_subject2']['dof'].shape)
print(data['g1_dance1_subject2']['root_rot'].shape)