import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import umap


def scatterplot_lab(embedding, y, ax, label=None, **kwargs):
    l = np.unique(y)
    label = label or l
    scatter = [ax.scatter(
        embedding[y == v, 0],
        embedding[y == v, 1],
        #c=[sns.color_palette()[v] for v in l])
        label=l,
        **kwargs
    ) for l, v in zip(label, l)]
    #axs[0].set_aspect('equal', 'datalim')
    return scatter


def scatterplot(embedding, y, ax, **kwargs):
    l = np.unique(y)
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=y,
        vmax=9,
        cmap=ListedColormap(sns.color_palette()),
        **kwargs)
    #axs[0].set_aspect('equal', 'datalim')
    return scatter


dataset = "mnist"
#dataset = "penguins"
if dataset == "mnist":
    mnist = datasets.fetch_openml("mnist_784", parser='auto')
    X = mnist.data.values
    X = X/X.max()
    targets = mnist.target.cat.categories
    targets = dict(zip(targets, range(len(targets))))
    y = mnist.target.map(targets)
    tt_split = lambda x: (x[:60000], x[60000:])
else:
    penguins = pd.read_csv(
        "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins.csv")
    penguins = penguins.dropna()
    penguins['sex'] = penguins['sex'].astype("category").cat.codes
    penguins['island'] = penguins['island'].astype("category").cat.codes

    sns.pairplot(penguins.drop("year", axis=1), hue='species')

    X = penguins.select_dtypes(float).values
    #X = penguins.drop(["year", "island"], axis=1).select_dtypes("number").values
    X = StandardScaler().fit_transform(X)

    #targets = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
    targets = np.unique(penguins.species)
    targets = dict(zip(targets, range(len(targets))))
    y = penguins.species.map(targets)


print("Linear separability of full dataset (no holdout) via Linear Discriminant Analysis")
yh = LinearDiscriminantAnalysis(n_components=2).fit(X, y=y).predict(X)
print(pd.DataFrame(confusion_matrix(y, yh), index=list(targets), columns=list(targets)))
print(classification_report(y, yh, labels=list(targets.values()), target_names=list(targets)))


n_components = 2
n_neighbors = 15  # def. 15
min_dist = 0.1  # def. 0.1
scatter_kwargs = dict(s=5, alpha=.6)

axs = plt.subplots(1, 4, sharex=True, sharey=True)[1]
i = 0

Xp = PCA(n_components=2).fit_transform(X)
l = scatterplot(Xp, y=y, ax=axs[i], **scatter_kwargs)
axs[i].legend(l.legend_elements()[0], list(targets))
axs[i].set_title('PCA')

i += 1
Xl = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y=y)
scatterplot(Xl, y=y, ax=axs[i], **scatter_kwargs)
axs[i].set_title('LDA')

i += 1
reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist)
embedding = reducer.fit_transform(X)

l = scatterplot(embedding, y=y, ax=axs[i], **scatter_kwargs)
axs[i].legend(l.legend_elements()[0], list(targets))
axs[i].set_title('UMAP')

i += 1
t_sne = manifold.TSNE(
    n_components=n_components,
    perplexity=15,
    init="random",
    n_iter=255,  #
    random_state=0,
)
S_t_sne = t_sne.fit_transform(X)
scatterplot(S_t_sne, y=y, ax=axs[i], **scatter_kwargs)
axs[i].set_title(f'T-SNE')

if len(X) < 1000000:
    t_sne = manifold.TSNE(
        n_components=n_components,
        perplexity=15,
        init="random",
        n_iter=1000,
        random_state=0,
    )
    S_t_sne = t_sne.fit_transform(X)
    scatterplot(S_t_sne, y=y, ax=plt.subplots()[1], **scatter_kwargs)
    plt.title(f'T-SNE: perplexity {t_sne.perplexity} n_iter {t_sne.n_iter}')

i += 1
plt.show()
