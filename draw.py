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

#plt.figure(figsize=(16, 8))

def draw_line_mlu(data,b_show=False):
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
    if b_show:
        plt.show()
    plt.savefig(f'{singleton.models_path}/mlu.png')
    plt.close()


def draw_line_sum(data,b_show=False):
    plt.clf()
    for i,key in enumerate(data.keys()):
        color = cmap(i)
        x = range(len(data[key]))
        plt.plot(x, data[key], color=color, linestyle=line_styles[i], label=key)

    plt.title('sum')
    plt.xlabel('tm_idx')
    plt.ylabel('sum')
    plt.legend(loc='best')
    if b_show:
        plt.show()
    plt.savefig(f'{singleton.models_path}/sum.png')
    plt.close()

def draw_box_mlu(data,b_show=False):
    plt.clf()
    vals = data.values()
    keys = data.keys()
    plt.boxplot(vals, labels=keys)
    plt.title("Multiple Box Plots Comparison")
    plt.xlabel("Data Groups")
    plt.ylabel("Value")
    plt.legend()
    if b_show:
        plt.show()
    plt.savefig(f'{singleton.models_path}/box_mlu.png')
    plt.close()

def draw_cdf_cdf(data,b_show=False):
    plt.clf()

    for i,key in enumerate(data.keys()):
        sorted_data1 = np.sort(data[key])
        n1 = len(sorted_data1)
        cdf = np.arange(1, n1 + 1) / n1
        plt.plot(sorted_data1, cdf, label=key)

    plt.title("CDF")
    plt.xlabel("Data Value")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    if b_show:
        plt.show()
    plt.savefig(f'{singleton.models_path}/cdf_mlu.png')
    plt.close()




def draw_box_sum(data,b_show=False):
    plt.clf()
    vals = data.values()
    keys = data.keys()
    plt.boxplot(vals, labels=keys)
    plt.title("Multiple Box Plots Comparison")
    plt.xlabel("Data Groups")
    plt.ylabel("Sum")
    plt.legend()
    if b_show:
        plt.show()
    plt.savefig(f'{singleton.models_path}/box_sum.png')
    plt.close()

def draw_cdf_sum(data,b_show=False):
    plt.clf()

    for i,key in enumerate(data.keys()):
        sorted_data1 = np.sort(data[key])
        n1 = len(sorted_data1)
        cdf = np.arange(1, n1 + 1) / n1
        plt.plot(sorted_data1, cdf, label=key)

    plt.title("CDF")
    plt.xlabel("Data Value")
    plt.ylabel("Sum Cumulative Probability")
    plt.legend()
    if b_show:
        plt.show()
    plt.savefig(f'{singleton.models_path}/cdf_sum.png')
    plt.close()


def draw_heatmap(key,data,b_show=False,vamx = 0.10):
    plt.clf()
    graph = np.zeros((singleton.num_nodes, singleton.num_nodes))

    for idx,load in enumerate(data):
        s,d = singleton.idx2edge[idx]
        graph[s][d] = load
    
    g = sns.heatmap(graph, cmap='viridis',vmin=0, vmax=vamx)
    if b_show:
        plt.show()
    plt.savefig(f'{singleton.models_path}/heatmap_{key}.png')
    plt.close()


def draw_2dheatmap(key,data,b_show=False):
    plt.clf()
    g = sns.heatmap(data, cmap='viridis',vmin=0, vmax=1)
    if b_show:
        plt.show()
    plt.savefig(f'{singleton.models_path}/heatmap_{key}.png')
    plt.close()
        
