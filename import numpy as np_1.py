import numpy as np
import pandas as pd

print(np.load('filtered_distances.npy'))
pd.DataFrame(np.load('filtered_distances.npy')).to_csv('filtered_distances.csv', index=False)
