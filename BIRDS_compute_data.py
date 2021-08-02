import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import csv
import h5py
from tqdm import tqdm
import re

import warnings
warnings.filterwarnings('ignore')


from analysis_yamnet import (lineplot, add_in_weekend_shading, load_audio_data, compute_presence, 
                     presence_change_percentage, add_aligned_hour_index,
                     calculate_rolling_change_in_value)
# DATAPATH = '/scratch/mf3734/covid-related-analysis/data/'
# DATAPATH = '/beegfs/mc6591/covid/'
DATAPATH = '/scratch/mf3734/share/arise/data/'
AUDIO_DATAPATH = os.path.join(DATAPATH, "yamnet/")
NODE_INFOPATH = 'sonyc_node_info.csv'
BIRD_SAMPLE_PATH = 'filelist.csv'
bird_related_classes = np.arange(106, 116)
bird_related_classes
classes_names = ['Bird', 'Bird vocalization, bird call, bird song', 'Chirp, tweet', 'Squawk', 'Pigeon, dove',
                 'Coo', 'Crow', 'Caw', 'Owl', 'Hoot']
important_dates = {
    'NYU classes online': datetime.date(2020, 3, 11),
    'Bars, restaurants closed': datetime.date(2020, 3, 16),
    'Non-essential bussines and housing closed': datetime.date(2020, 3, 22),
}
sensors_wsp = ['b827eb815321', 'b827eb0fedda', 'b827eb905497', 'b827eb8e2420']
sensors_wsp
pat = '(.+)_(\d+\.\d\d)'
prog = re.compile(pat)
files = os.listdir(AUDIO_DATAPATH)
for f in files:
    if "h5" in f:
        print(f)
        filepath = os.path.join(AUDIO_DATAPATH, f)
        h5 = h5py.File(filepath, 'r')
        df_ = pd.DataFrame({})
        if f.split('_')[0] in sensors_wsp:
            for key in tqdm(h5.keys()):
                _dict = {'sensor_id': prog.match(key).group(1),
                         'timestamp': float(prog.match(key).group(2)),
                         'filename': f}
                _dict.update({str(c): h5[key][:, c] for c in bird_related_classes})
                df_ = pd.concat((df_, pd.DataFrame(_dict)))

            df_.to_pickle(os.path.join(AUDIO_DATAPATH, f'dataframe_yamnet_{prog.match(key).group(1)}.pkl'))
            # df = pd.read_hdf(filepath)
