import os

DIR = '/datatmp2/joan/tfg_joan/videos/bench_press'

i = 1

for file in sorted(os.listdir(DIR)):
    new_name = fr'train_bench_press_{i:03d}.mp4'
    os.rename(os.path.join(DIR, file), os.path.join(DIR, new_name))
    i += 1
    
