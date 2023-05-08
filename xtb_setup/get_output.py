#!/groups/kemi/koerstz/anaconda3/envs/quantum_ml/bin/python

import pickle
import pandas as pd
import numpy as np
import sys
from rdkit import Chem
from rdkit.Chem import Descriptors
sys.path.append("/home/jelholm/opt/tQMC/QMC")
import qmconf


df = pd.read_pickle(sys.argv[1])
print(df)
inp = input("Do you want to get .xyz files for the systems? y/n" )
if inp.lower() == "y":
    for x in df.itertuples():
        x.monos.write_xyz(to_file=True)
        x.reac.write_xyz(to_file=True)
        x.prod.write_xyz(to_file=True)
        x.scan_ts.write_xyz(to_file=True)
        print(x.monos.results['energy'])
