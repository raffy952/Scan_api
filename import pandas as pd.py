import pandas as pd
import numpy as np


print(np.load('scans/scan_0000.npz', allow_pickle=True)['distances'])
print(np.load('scans/scan_0000.npz', allow_pickle=True)['theta'])
