import numpy as np
import os
import random
from .vectors import angle_between


def find_closest_chunk(features, data):
    if isinstance(data, np.ndarray):
        all_chunks = data
    else:
        all_chunks = np.load(os.path.join(data, 'Y_all.npy'))
    chunks_feats = all_chunks[:, 2:]

    features = features[0]

    distances = list(map(lambda chunk: angle_between(chunk, features),
                         chunks_feats))

    min_val = min(distances)
    closest_indices = [i for i, n in enumerate(distances) if n == min_val]
    closest_index = random.choice(closest_indices)
    closest = all_chunks[closest_index, :]

    return closest[0], closest[2:]
