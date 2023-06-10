import matplotlib.pyplot as plt
from clustimage import Clustimage
import os
import numpy as np
import optuna
import copy
from tqdm import tqdm
import pickle

# init with HOG method
cl = Clustimage(method='hog', params_hog={'pixels_per_cell':(4,4)})

X = []
path = "data/train/cloth-mask/"
pathnames = os.listdir(path)
for i in tqdm(pathnames):
    X.append(cl.imread(f'{path}/{i}', dim=(128,128), colorscale=0,flatten=True))
X = np.array(X)

X_img = cl.import_data(X)
Xfeat = cl.extract_feat(X_img)
xycoord = cl.embedding(Xfeat)
agglo_clust = copy.deepcopy(cl)
agglo_clust.cluster(cluster='agglomerative', evaluate='silhouette')

def make_files(p,name):
    pathnames = os.listdir(p)
    arr = {}
    k = None
    for i in pathnames:
        try:
            img = cl.imread(f'{p}/{i}', dim=(128,128), colorscale=0,flatten=True)
            result = agglo_clust.find(img)

            f = [*result.keys()][1]
            k = result[f]
            clust_id = k['labels'][0]
            if clust_id not in arr:
                arr[clust_id] = [i]
            else:
                arr[clust_id].append(i)
        except:
            print(i)

    with open(f'file2_{name}.pkl', 'wb') as f:
        pickle.dump(arr, f)
    with open(f'file2_{name}.pkl', 'rb') as f:
        clusts = pickle.load(f)
    print(os.getcwd())
    print(os.listdir())
    # os.mkdir('data/data3')
    for i in clusts:
        f = open(f'data/data3/{name}{i}.txt','w')
        for j in clusts[i]:
            f.write(f'{j} {j}\n')
        f.close()


make_files("data/test/cloth-mask/", 'test')
make_files("data/train/cloth-mask/", 'train')
print(os.getcwd())
print(os.listdir())
