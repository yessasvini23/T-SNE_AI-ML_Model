from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
             discriminant_analysis, random_projection)
step=3
if(step>0):
    ## Loading and curating the data
    digits = datasets.load_digits(n_class=10)
    X = digits.data
    y = digits.target
    digits.data.shape
    #print(digits['DESCR'])
    nrows, ncols = 2, 5
    plt.figure(figsize=(6,3))
    plt.gray()
    for i in range(ncols * nrows):
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.matshow(digits.images[i,...])
        plt.xticks([]); plt.yticks([])
        plt.title(digits.target[i])
    plt.savefig('images/digits-generated.png', dpi=150)
    plt.show()
n_samples, n_features = X.shape
n_neighbors = 30

## Function to Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)     
    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        ## only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                ## don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

#----------------------------------------------------------------------
if(step>0):
## Plot images of the digits
    n_img_per_row = 20
    img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
    for i in range(n_img_per_row):
        ix = 10 * i + 1
        for j in range(n_img_per_row):
            iy = 10 * j + 1
            img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title('A selection from the 64-dimensional digits dataset')
    plt.show()
if(step>1):
    ## Computing PCA
    print("Computing PCA projection")
    t0 = time()
    X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    plot_embedding(X_pca,
                   "Principal Components projection of the digits (time %.2fs)" %
                   (time() - t0))
    plt.show()
if(step>2):
    ## Computing t-SNE
    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, perplexity=5, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne,
                   "t-SNE embedding of the digits (perplexity=5, time %.2fs)" %
                   (time() - t0))
    plt.show()
    tsne = manifold.TSNE(n_components=2, perplexity=15, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne,
                   "t-SNE embedding of the digits (perplexity=15, time %.2fs)" %
                   (time() - t0))
    plt.show()
    tsne = manifold.TSNE(n_components=2, perplexity=25, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne,
                   "t-SNE embedding of the digits (perplexity=25, time %.2fs)" %
                   (time() - t0))
    plt.show()
    tsne = manifold.TSNE(n_components=2, perplexity=40, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne,
                   "t-SNE embedding of the digits (perplexity=40, time %.2fs)" %
                   (time() - t0))
    plt.show()
    tsne = manifold.TSNE(n_components=2, perplexity=50, init='pca', random_state=0)
    t0 = time()
    X_tsne = tsne.fit_transform(X)
    plot_embedding(X_tsne,
                   "t-SNE embedding of the digits (perplexity=50, time %.2fs)" %
                   (time() - t0))
    plt.show()
