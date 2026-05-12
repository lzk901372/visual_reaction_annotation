import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from feat import Detector
import pandas as pd

from glob import glob
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


KEYPOINT_LIST = [
    'frame',
    'approx_time',
    'x_0',
     'y_0',
     'x_1',
     'y_1',
     'x_2',
     'y_2',
     'x_3',
     'y_3',
     'x_4',
     'y_4',
     'x_5',
     'y_5',
     'x_6',
     'y_6',
     'x_7',
     'y_7',
     'x_8',
     'y_8',
     'x_9',
     'y_9',
     'x_10',
     'y_10',
     'x_11',
     'y_11',
     'x_12',
     'y_12',
     'x_13',
     'y_13',
     'x_14',
     'y_14',
     'x_15',
     'y_15',
     'x_16',
     'y_16',
     'x_17',
     'y_17',
     'x_18',
     'y_18',
     'x_19',
     'y_19',
     'x_20',
     'y_20',
     'x_21',
     'y_21',
     'x_22',
     'y_22',
     'x_23',
     'y_23',
     'x_24',
     'y_24',
     'x_25',
     'y_25',
     'x_26',
     'y_26',
     'x_27',
     'y_27',
     'x_28',
     'y_28',
     'x_29',
     'y_29',
     'x_30',
     'y_30',
     'x_31',
     'y_31',
     'x_32',
     'y_32',
     'x_33',
     'y_33',
     'x_34',
     'y_34',
     'x_35',
     'y_35',
     'x_36',
     'y_36',
     'x_37',
     'y_37',
     'x_38',
     'y_38',
     'x_39',
     'y_39',
     'x_40',
     'y_40',
     'x_41',
     'y_41',
     'x_42',
     'y_42',
     'x_43',
     'y_43',
     'x_44',
     'y_44',
     'x_45',
     'y_45',
     'x_46',
     'y_46',
     'x_47',
     'y_47',
     'x_48',
     'y_48',
     'x_49',
     'y_49',
     'x_50',
     'y_50',
     'x_51',
     'y_51',
     'x_52',
     'y_52',
     'x_53',
     'y_53',
     'x_54',
     'y_54',
     'x_55',
     'y_55',
     'x_56',
     'y_56',
     'x_57',
     'y_57',
     'x_58',
     'y_58',
     'x_59',
     'y_59',
     'x_60',
     'y_60',
     'x_61',
     'y_61',
     'x_62',
     'y_62',
     'x_63',
     'y_63',
     'x_64',
     'y_64',
     'x_65',
     'y_65',
     'x_66',
     'y_66',
     'x_67',
     'y_67',]

LIST_TO_SAVE = ['frame',
     'approx_time',
     'Pitch', 
     'Roll', 
     'Yaw', 
     'AU04',
    #  'anger',
    #  'disgust',
    #  'fear',
    #  'happiness',
    #  'sadness',
    #  'surprise',
    #  'neutral',
     ]

# 1. 初始化检测器 (自动加载模型)
# au_model 可选 'svm' 或 'jaanet'，emotion_model 可选 'resmasknet'
detector = Detector(
    face_model="retinaface",
    au_model="xgb", 
    emotion_model="resmasknet",
    facepose_model="img2pose",
    device="cuda",
    n_jobs=4,
    no_pose=False,
    no_identity=True,
    no_emotion=True,
)

# # 2. 检测视频
# # input_video 是你的视频路径
# input_video_path = "/data/zikai/Data/RealTalkListening/Original/ZV9K28EmGIE/0000.mp4"
# output_root_dir = "output"
# os.makedirs(output_root_dir, exist_ok=True)
# video_output_subdir = os.path.join(output_root_dir, Path(input_video_path).parent.name)
# os.makedirs(video_output_subdir, exist_ok=True)

# video_prediction = detector.detect_video(
#     input_video_path, 
#     skip_frames=None,
#     output_size=512,
#     batch_size=64,
#     num_workers=4,
#     no_pose=False,
#     no_identity=True,
#     no_emotion=True,
# )

# # 3. 提取关键数据
# # video_prediction 是一个包含每一帧数据的 DataFrame
# video_prediction = video_prediction.replace(r'^\s*$', np.nan, regex=True)
# video_prediction = video_prediction.dropna(subset=LIST_TO_SAVE, how='any')

# general_results = video_prediction.loc[:, LIST_TO_SAVE].copy()
# keypoint_results = video_prediction.loc[:, KEYPOINT_LIST].copy()

# # # 4. 导出 CSV
# # # 每一行代表一帧，AU04 列就是皱眉的置信度/强度
# output_general_csv_path = os.path.join(video_output_subdir, "head_and_frown.csv")
# output_keypoint_csv_path = os.path.join(video_output_subdir, "keypoints.csv")
# general_results.to_csv(output_general_csv_path, index=False)
# keypoint_results.to_csv(output_keypoint_csv_path, index=False)



'''
当皱眉时，AU04会变高，happy值会变低。可以按照这个规律去提取皱眉时间点。
'''

def get_output_paths(video_path: Path, output_root: Path):
    # print(type(video_path))
    csv_dir = output_root / video_path.parent.stem / video_path.stem
    return {
        "csv_dir": csv_dir,
        "general_csv": csv_dir / "head_and_frown.csv",
        "keypoint_csv": csv_dir / "keypoints.csv",
    }



video_root_dir = "/data/zikai/Data/RealTalkListening/Original"
output_root = "output"
os.makedirs(output_root, exist_ok=True)

all_videos = sorted(glob(os.path.join(video_root_dir, "**/*.mp4"), recursive=True))
pending_videos = []
for p in all_videos:
    out = get_output_paths(Path(p), Path(output_root))
    if not (out["general_csv"].exists() and out["keypoint_csv"].exists()):
        pending_videos.append(p)

for video_path in tqdm(pending_videos[:1]):
    out = get_output_paths(Path(video_path), Path(output_root))
    out["csv_dir"].mkdir(parents=True, exist_ok=True)

    video_prediction = detector.detect_video(
        video_path, 
        skip_frames=None,
        output_size=512,
        batch_size=64,
        num_workers=4,
        no_pose=False,
        no_identity=True,
        no_emotion=True,
    )

    video_prediction = video_prediction.replace(r'^\s*$', np.nan, regex=True)
    video_prediction = video_prediction.dropna(subset=LIST_TO_SAVE, how='any')

    general_results = video_prediction.loc[:, LIST_TO_SAVE].copy()
    keypoint_results = video_prediction.loc[:, KEYPOINT_LIST].copy()

    general_results.to_csv(out["general_csv"], index=False)
    keypoint_results.to_csv(out["keypoint_csv"], index=False)

    print(f"Processed {video_path}")