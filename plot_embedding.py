from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# function to plot dataset embeddings
def plot_embedding(x, y, d, title=None):
    plt.figure(figsize=(7, 7), frameon=False)
    plt.subplot(111)
    for i in range(x.shape[0]):
        plt.text(x[i, 0], x[i, 1], str(y[i]),
                 color=plt.cm.bwr(d[i] / 1.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([]), plt.yticks([])
    plt.axis('off')
    if title is not None:
        plt.title(title)

# normalize image between 0 and 1
def norm_img(x):
    return ((x - np.min(x))) / (np.max(x) - np.min(x))

# flatten depthwise with mean
xt_train = np.mean(Xt_train, axis=1).reshape((Xt_train.shape[0], -1))[:500]
xs_train = np.mean(Xs_train, axis=1).reshape((Xs_train.shape[0], -1))[:500]

# find 2-dimensional embedding
tsne = TSNE(n_components=2, init='pca', random_state=0)
xs_tsne = norm_img(tsne.fit_transform(xs_train))
xt_tsne = norm_img(tsne.fit_transform(xt_train))

# prepare data
x_tsne = np.concatenate((xs_tsne, xt_tsne)) 
class_lab = np.concatenate((ys_train[:500].argmax(1), yt_train[:500].argmax(1)))
domain_lab = np.concatenate((np.repeat(0, len(xs_tsne)), np.repeat(1, len(xt_tsne))))

# plot the embeddings
plot_embedding(x_tsne, class_lab, domain_lab)
