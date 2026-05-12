import os
from tqdm import tqdm

txt_path = "usable_seamless.txt"
new_txt_path = "usable_seamless_2.txt"

with open(new_txt_path, "r") as f:
    lines = f.readlines()
    print(len(lines))
exit()

# with open(txt_path, "r") as f:
#     lines = f.readlines()

# new_lines = []
# for line in tqdm(lines):
#     parts = line.split(",")
#     video_path = parts[0]
#     fer_csv_path = parts[1]
#     pkl_path = parts[2]
    
#     new_fer_csv_path = fer_csv_path.replace("Seamless", "RealTalkListening/reaction_detect/fer_emotion/output_seamless")
#     new_pkl_path = pkl_path.replace("Seamless", "RealTalkListening/reaction_detect/nod_headshake/face_detection/output_seamless")

#     new_lines.append(f"{video_path},{new_fer_csv_path},{new_pkl_path}")

# with open(new_txt_path, "w") as f:
#     f.writelines(new_lines)