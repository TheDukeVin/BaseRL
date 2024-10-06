




from comp import *
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import pandas as pd
import time

np.random.seed(42)

def displayProjections(projection_mode, color_mode, player=0, displayMode=True):
    A = read_space_separated_file("hidden.out")
    x, y = A.shape
    firstPlayer = A[list(range(player, x, 2)), :]

    # Use smaller data set
    firstPlayer = firstPlayer[:10000, :]

    # Filter only X-player moves
    # firstPlayer = firstPlayer[firstPlayer[:, -5] == '0', :]

    firstPlayer_X = firstPlayer[:, :-5].astype(float)
    points = set()
    indices = []
    for i, a in enumerate(firstPlayer_X):
        if tuple(a) not in points:
            indices.append(i)
            points.add(tuple(a))
    
    firstPlayer = firstPlayer[indices, :]
    firstPlayer_X = firstPlayer[:, :-5].astype(float)

    if projection_mode == "TSNE":
        load_mode = "WRITE"
        if load_mode == "WRITE":
            start = time.time()
            embedded = TSNE(n_components=2, init='random', perplexity=10, random_state=42).fit_transform(firstPlayer_X)
            np.savetxt("tsne.csv", embedded)
            end = time.time()
            print(end - start)
        elif load_mode == "READ":
            embedded = np.loadtxt("tsne.csv")
    elif projection_mode == "PCA":
        embedded = PCA(n_components=2).fit_transform(firstPlayer_X)
    
    # firstPlayer_X = firstPlayer_X[:1000, :]
    # embedded = embedded[:1000]
    firstPlayer_state = firstPlayer[:, -4].astype(str)
    firstPlayer_timer = firstPlayer[:, -3].astype(float)
    firstPlayer_value = firstPlayer[:, -1].astype(float)

    names = firstPlayer_state

    cmap = plt.cm.RdYlGn

    fig,ax = plt.subplots(figsize=(8, 8))

    states = {}
    for i, a in enumerate(np.unique(firstPlayer_state)):
        states[a] = i
    state_array = []
    for i in firstPlayer_state:
        state_array.append(states[i])

    if color_mode == "STATE":
        c = np.array(state_array)
    elif color_mode == "VALUE":
        c = firstPlayer_value
    elif color_mode == "TIMER":
        c = firstPlayer_timer
    elif color_mode == "KMEANS":
        kmeans = KMeans(n_clusters=10, n_init=10).fit(firstPlayer_X).labels_
        c = kmeans

    sc = ax.scatter(embedded[:, 0], embedded[:, 1], c=c, s=5, cmap=cmap)

    annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        text = names[ind["ind"][0]]
        text = text[0:3].replace('.', '  ') + '\n' + text[3:6].replace('.', '  ') + '\n' + text[6:].replace('.', '  ')
        # text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
        #                     " ".join([names[n] for n in ind["ind"]]))
        annot.set_text(text)
        # annot.get_bbox_patch().set_facecolor(cmap(c[ind["ind"][0]]))
        annot.get_bbox_patch().set_alpha(0.8)
        

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

    fig.tight_layout()
    if displayMode:
        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.show()
    else:
        plt.savefig("img/" + projection_mode.lower() + "_" + color_mode.lower())

    # fig, ax = plt.subplots()
    # for timer in np.unique(firstPlayer_timer):
    #     ax.scatter(embedded[firstPlayer_timer == timer, 0], embedded[firstPlayer_timer == timer, 1], label=timer, s=2)
    # ax.legend()
    # plt.savefig("img/by_timer")

    # fig, ax = plt.subplots()
    # for clusterID in np.unique(cluster):
    #     ax.scatter(embedded[cluster == clusterID, 0], embedded[cluster == clusterID, 1], label=clusterID, s=2)
    # ax.legend()
    # plt.savefig("img/by_cluster")

if __name__ == '__main__':
    # for proj in ["TSNE", "PCA"]:
    #     for val in ["STATE", "VALUE", "TIMER", "KMEANS"]:
    #         displayProjections(proj, val, displayMode=False)

    displayProjections("TSNE", "STATE", player=0)


    # X = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0], [1, 1, 1]])
    # # Random init with same initialization for same data points
    # X_embedded = TSNE(n_components=2, learning_rate='auto',
    #                 init=X[:,:2].astype('float'), perplexity=3).fit_transform(X)
    # print(X_embedded)