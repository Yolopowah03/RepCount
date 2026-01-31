import kagglehub # type: ignore
import os

DATASET_ONLINE_PATH1 = 'philosopher0808/gym-workoutexercises-video'
DOWNLOAD_PATH1 = '/datatmp2/joan/repCount/videos_new/'

DATASET_ONLINE_PATH2 = 'philosopher0808/gym-workoutexercises-video'
DOWNLOAD_PATH2 = '/datatmp2/joan/repCount/videos_new_2'

DATASET_ONLINE_PATH3 = 'philosopher0808/gym-workoutexercises-video'
DOWNLOAD_PATH3 = '/datatmp2/joan/repCount/videos_final'

DATASET_ONLINE_PATH4 = 'philosopher0808/gym-workoutexercises-video'
DOWNLOAD_PATH4 = '/datatmp2/joan/repCount/videos_final'

DATASET_ONLINE_PATH5 = 'philosopher0808/gym-workoutexercises-video'
DOWNLOAD_PATH5 = '/datatmp2/joan/repCount/videos_final'

DATASET_ONLINE_PATH6 = 'philosopher0808/gym-workoutexercises-video'
DOWNLOAD_PATH6 = '/datatmp2/joan/repCount/videos_final'

DATASET_ONLINE_PATH7 = 'philosopher0808/gym-workoutexercises-video'
DOWNLOAD_PATH7 = '/datatmp2/joan/repCount/videos_final'

DATASET_ONLINE_PATH8 = 'philosopher0808/gym-workoutexercises-video'
DOWNLOAD_PATH8 = '/datatmp2/joan/repCount/videos_final'

os.makedirs(DOWNLOAD_PATH1, exist_ok=True)
os.makedirs(DOWNLOAD_PATH2, exist_ok=True)
os.makedirs(DOWNLOAD_PATH3, exist_ok=True)
os.makedirs(DOWNLOAD_PATH4, exist_ok=True)
os.makedirs(DOWNLOAD_PATH5, exist_ok=True)
os.makedirs(DOWNLOAD_PATH6, exist_ok=True)
os.makedirs(DOWNLOAD_PATH7, exist_ok=True)
os.makedirs(DOWNLOAD_PATH8, exist_ok=True)

def download_kaggle_dataset(dataset_name: str, download_path: str) -> None:
    """
    Downloads a dataset from Kaggle using the kagglehub library.

    Parameters:
    - dataset_name (str): The name of the Kaggle dataset to download (e.g., 'username/dataset-name').
    - download_path (str): The local path where the dataset should be downloaded.
    """
    kagglehub.download_dataset(dataset_name, download_path)

download_kaggle_dataset('username/dataset-name', path='./data/dataset')