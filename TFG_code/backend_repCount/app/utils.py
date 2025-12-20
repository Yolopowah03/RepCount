import os
from .config import TEMP_UPLOAD_PATH, TEMP_OUTPUT_PATH

def clean_files():
    
    print("Netejant arxius...")
    
    dirs_to_clean = [TEMP_UPLOAD_PATH, TEMP_OUTPUT_PATH]
    
    for directory in dirs_to_clean:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            
            if os.path.isfile(filepath) and "thumbnail" not in filename:
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Error al eliminar {filename}: {e}")