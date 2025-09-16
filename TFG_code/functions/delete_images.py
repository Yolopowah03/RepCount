import os

for file in os.listdir('/datatmp2/joan/tfg_joan/images/train/squat'):
    if not file.endswith('_0000.jpg'):
        os.remove(os.path.join('/datatmp2/joan/tfg_joan/images/train/squat', file))