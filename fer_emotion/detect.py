import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from fer.classes import Video
from fer.fer import FER
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from glob import glob
from tqdm import tqdm
from pathlib import Path


video_root_dir = "/data/zikai/Data/RealTalkListening/Original"
video_paths = glob(os.path.join(video_root_dir, "**/*.mp4"), recursive=True)
# video_paths = [p for p in video_paths if os.path.isfile(p)]
video_paths = ["/data/zikai/Data/RealTalkListening/Original/-1wCGAxqT_4/0001.mp4", ]

detector = FER(mtcnn=True) # mtcnn=True 精度更高

for video_path in tqdm(video_paths):
    subdir = Path(video_path).parent.stem
    video_name = Path(video_path).stem
    video = Video(video_path, dataset_name="RealTalkListening")
    raw_data = video.analyze(
        detector, 
        display=False, 
        output=None, 
        save_frames=False, 
        save_video=True, 
        annotate_frames=True, 
        zip_images=False, 
        detection_box=None, 
        lang="en", 
        include_audio=False,
        size_multiplier=1,
        batch_size=64,
        use_async_io=True,
    )
    df = video.to_pandas(raw_data)

    csv_dir = os.path.join('output', subdir, 'emo_scores')
    os.makedirs(csv_dir, exist_ok=True)
    df.to_csv(os.path.join(csv_dir, f"{video_name}.csv"), index=False)

# # 初始化检测器
# detector = FER(mtcnn=True) # mtcnn=True 精度更高
# video_path = "/data/zikai/Data/RealTalkListening/Original/_0VwR9WPS-k/0003.mp4"
# subdir = Path(video_path).parent.stem
# video_name = Path(video_path).stem
# video = Video(video_path, dataset_name="RealTalkListening")

# # 分析视频，每 1 帧分析一次（可调频率）
# raw_data = video.analyze(
#     detector, 
#     output=None,
#     display=False,
#     save_frames=False,
#     save_video=True,
#     annotate_frames=True,
#     zip_images=False,
#     detection_box=None,
#     lang="en",
#     include_audio=False,
# )

# # 转换为 Pandas DataFrame 并导出 CSV
# csv_dir = os.path.join('output', subdir, 'emo_scores')
# os.makedirs(csv_dir, exist_ok=True)
# df = video.to_pandas(raw_data)
# df.to_csv(os.path.join(csv_dir, f"{video_name}.csv"), index=False)