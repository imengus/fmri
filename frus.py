import os
import re
import glob
import numpy as np
import pandas as pd
import scipy
from statsmodels.stats.multitest import fdrcorrection
import nilearn

from numba import jit

def main():
    cwd = '/home/ilkin/Documents/GitHub/datasci/fmri/'
    directory = cwd + 'difumo_atlases/1024/2mm/binarised_rois'  
    rois = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            rois.append(int(re.findall('_(.*?)\.', filename)[0]))
    rois = np.array(sorted(rois))
    n_parcs = len(rois)

    @jit
    def ret_frus(mat):
        frus_mat = np.zeros(n_parcs)
        for i in range(n_parcs):
            for j in range(n_parcs):
                for k in range(n_parcs):
                    if (i != j) and (i != k) and (j != k):
                        y = mat[i,j]*mat[i,k]*mat[j,k] < 0
                        frus_mat[i] += y
        return frus_mat / 6

    d = {}
    for mod in ['LSD', 'PCB']:
        d[mod] = rois.copy()
        for rest in ['rest1', 'rest2', 'rest3']:
            for sub in sorted(glob.glob('/home/ilkin/Documents/GitHub/datasci/fmri/' + f'difumo/{mod}/{rest}/*all*')):
                mat = pd.read_csv(sub, delimiter=' ', header=None)[rois]
                corr = np.corrcoef(mat.to_numpy().T)
                frus = ret_frus(corr)
                d[mod] = np.vstack((d[mod], frus))
            print(mod, rest, "complete")
            
        m = np.repeat(np.mean(d[mod], axis=1, keepdims=True), n_parcs, axis=1)
        s = np.repeat(np.std(d[mod], axis=1, keepdims=True), n_parcs, axis=1)
        d[mod] = (d[mod] - m) / s
        np.save(mod, d[mod])

    p_values = []
    for col1, col2 in zip(d[mod].T, d[mod].T):
        _, p_value = scipy.stats.ttest_rel(col1, col2)
        p_values.append(p_value)
    rejected, adjusted_p_values = fdrcorrection(p_values)

    difumo = nilearn.datasets.fetch_atlas_difumo(dimension=1024)
    regions = difumo.labels[rois[rejected] - 1]
    with open('regions.txt', 'w+') as f:
        f.write(regions)

if __name__ == "__main__":
    main()