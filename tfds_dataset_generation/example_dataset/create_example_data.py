import numpy as np
import tqdm
import os
from utils import DATA_DIR

N_TRAIN_EPISODES = 10
EPISODE_LENGTH = 10


def create_fake_episode(path):
    episode = []
    for step in range(EPISODE_LENGTH):
        episode.append({
            'image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
            'wrist_image': np.asarray(np.random.rand(64, 64, 3) * 255, dtype=np.uint8),
            'mask': np.asarray(np.random.rand(64, 64, 1) * 100, dtype=np.uint8),
            'wrist_mask': np.asarray(np.random.rand(64, 64, 1) * 100, dtype=np.uint8),
            'state': np.asarray(np.random.rand(10), dtype=np.float32),
            'action': np.asarray(np.random.rand(10), dtype=np.float32),
            'language_instruction': 'dummy instruction',
        })
    np.save(path, episode)


# create fake episodes for train and validation
print("Generating train examples...")
os.makedirs(DATA_DIR / 'raw_datasets/example_dataset/train', exist_ok=True)
for i in tqdm.tqdm(range(N_TRAIN_EPISODES)):
    create_fake_episode(DATA_DIR / f'raw_datasets/example_dataset/train/episode_{i}.npy')

print('Successfully created example data!')
