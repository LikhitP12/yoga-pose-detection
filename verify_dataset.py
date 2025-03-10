import os

# Path to dataset folder
data_dir = "data/"

# Check structure
if os.path.exists(data_dir):
    poses = os.listdir(data_dir)
    print("Yoga Poses Found:", poses)

    for pose in poses:
        pose_path = os.path.join(data_dir, pose)
        images = os.listdir(pose_path)
        print(f"{pose}: {len(images)} images")
else:
    print("Dataset folder not found. Please check the path.")