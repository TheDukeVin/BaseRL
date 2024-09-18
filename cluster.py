
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(1)

x = np.random.rand(15)
y = np.random.rand(15)
names = np.array(list("ABCDEFGHIJKLMNO"))
c = np.random.randint(1,5,size=15)

norm = plt.Normalize(1,4)
cmap = plt.cm.RdYlGn

fig,ax = plt.subplots()
sc = plt.scatter(x,y,c=c, s=100, cmap=cmap, norm=norm)

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):
    
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    annot.get_bbox_patch().set_alpha(0.4)
    

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()


# from comp import *
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA

# np.random.seed(42)

# def displayProjections():
#     A = read_space_separated_file("hidden.out")
#     x, y = A.shape
#     firstPlayer = A[list(range(0, x, 2)), :]
#     firstPlayer = firstPlayer[firstPlayer[:, -5] == '0', :]
#     firstPlayer_X = firstPlayer[:, :-5].astype(float)
#     firstPlayer_state = firstPlayer[:, -4].astype(str)
#     firstPlayer_timer = firstPlayer[:, -3].astype(str)
#     # embedded = TSNE(n_components=2).fit_transform(firstPlayer_X)
#     embedded = PCA(n_components=2).fit_transform(firstPlayer_X)
#     cluster = KMeans(n_clusters=8).fit(firstPlayer_X).labels_

#     fig, ax = plt.subplots()
#     for state in np.unique(firstPlayer_state):
#         ax.scatter(embedded[firstPlayer_state == state, 0], embedded[firstPlayer_state == state, 1], color=np.random.rand(3,), label=state, s=2)
#     ax.legend()
#     plt.savefig("img/by_state")

#     fig, ax = plt.subplots()
#     for timer in np.unique(firstPlayer_timer):
#         ax.scatter(embedded[firstPlayer_timer == timer, 0], embedded[firstPlayer_timer == timer, 1], label=timer, s=2)
#     ax.legend()
#     plt.savefig("img/by_timer")

#     fig, ax = plt.subplots()
#     for clusterID in np.unique(cluster):
#         ax.scatter(embedded[cluster == clusterID, 0], embedded[cluster == clusterID, 1], label=clusterID, s=2)
#     ax.legend()
#     plt.savefig("img/by_cluster")

# if __name__ == '__main__':
#     # displayProjections()
#     # A = np.array([
#     #     [1, 2, 3],
#     #     [3, 4, 5],
#     #     [1, 2, 3],
#     #     [2, 9, 7]
#     # ])
#     # embedded = TSNE(n_components=2, perplexity=1).fit_transform(A)
#     # print(embedded)
#     X = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0], [1, 1, 1]])
#     X_embedded = TSNE(n_components=2, learning_rate='auto',
#                     init='random', perplexity=3).fit_transform(X)
#     print(X_embedded)