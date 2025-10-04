import os

DIR = '/datatmp2/joan/tfg_joan/videos/train/squat'

i = 1

for file in sorted(os.listdir(DIR)):
    new_name = fr'train_squat_{i:03d}.mp4'
    os.rename(os.path.join(DIR, file), os.path.join(DIR, new_name))
    i += 1