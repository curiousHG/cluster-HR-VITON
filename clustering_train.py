import matplotlib.pyplot as plt
from clustimage import Clustimage
import os
import numpy as np
import optuna
import copy

# init with HOG method
cl = Clustimage(method='hog', params_hog={'pixels_per_cell':(4,4)})

# Load MNIST example data
# X = cl.import_example(data='mnist')
X = []
path = "data/train/cloth-mask/"
pathnames = os.listdir(path)
print(len(pathnames))

for i in pathnames:
    X.append(cl.imread(f'{path}/{i}', dim=(128,128), colorscale=0,flatten=True))
X = np.array(X)
dim=(128,128)
print(X)
X_img = cl.import_data(X)

Xfeat = cl.extract_feat(X_img)
# Embedding using tSNE
xycoord = cl.embedding(Xfeat)

agglo_clust = copy.deepcopy(cl)
# kmeans_clust = copy.deepcopy(cl)

agglo_clust.cluster(cluster='agglomerative', evaluate='silhouette')
# kmeans_clust.cluster(cluster='kmeans',evaluate='silhouette')
fig = agglo_clust.scatter(zoom=0.3,img_mean = True)
plt.savefig("Agglo_Clust")
# fig = kmeans_clust.scatter(zoom=0.3,img_mean = True)
# plt.savefig("Kmeans_Clust")

# agglo_result = agglo_clust.find(X[0:3])

pathnames = os.listdir(path)

arr = {}
k = None
for i in pathnames:
    img = cl.imread(f'{path}/{i}', dim=(128,128), colorscale=0,flatten=True)
    result = agglo_clust.find(img)

    f = [*result.keys()][1]
    k = result[f]
    # print(k['labels'])
    try:

        clust_id = k['labels'][0]

        if clust_id not in arr:
            arr[clust_id] = [i]
        else:
            arr[clust_id].append(i)
    except:
        print(i)
    # break


import pickle
with open('train_saved_dictionary.pkl', 'wb') as f:
    pickle.dump(arr, f)
        
with open('train_saved_dictionary.pkl', 'rb') as f:
    clusts = pickle.load(f)

for i in clusts:
    f = open(f'data/train{i}.txt','w')
    for j in clusts[i]:
        f.write(f'{j} {j}\n')
    f.close()