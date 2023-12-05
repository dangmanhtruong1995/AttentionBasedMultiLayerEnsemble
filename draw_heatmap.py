import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# a = np.random.random((16, 16))
arr = np.loadtxt("./result/artificial1/attention_normalized_layer_2.txt", delimiter=",")

# plt.imshow(arr, cmap='hot', interpolation='nearest')
# plt.show()

# mask = np.zeros_like(arr)
# mask[np.triu_indices_from(mask)] = True
# with sns.axes_style("white"):
    # ax = sns.heatmap(arr, mask=mask, vmax=.3, square=True,  cmap="YlGnBu")
    # plt.show()
    
def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()


# test_array = np.arange(100 * 100).reshape(100, 100)
heatmap2d(arr)