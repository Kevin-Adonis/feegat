import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
from top import singleton

print(sns.__version__)
print(matplotlib.__version__)
print(np.__version__)


line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 10)), (0, (1, 1))]
cmap = ListedColormap(plt.get_cmap('tab10')(np.linspace(0, 1, 7)))

def draw_mlu(data):
    plt.clf()
    for i,key in enumerate(data.keys()):
        color = cmap(i)
        x = range(len(data[key]))
        plt.plot(x, data[key], color=color, linestyle=line_styles[i], label=key)

    plt.title('mlu')
    plt.xlabel('tm_idx')
    plt.ylabel('mlu')
    # 显示图例，用于说明每条折线对应的标签，位置可以通过参数调整，
    # 这里设置为最佳位置（自动调整到不遮挡图表内容的合适地方）
    plt.legend(loc='best')
    plt.savefig(f'{singleton.models_path}/mlu.png')
    plt.close()


def draw_sum(data):
    plt.clf()
    for i,key in enumerate(data.keys()):
        color = cmap(i)
        x = range(len(data[key]))
        plt.plot(x, data[key], color=color, linestyle=line_styles[i], label=key)

    plt.title('sum')
    plt.xlabel('tm_idx')
    plt.ylabel('sum')
    plt.legend(loc='best')
    plt.savefig(f'{singleton.models_path}/sum.png')
    plt.close()