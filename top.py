import numpy as np
import tensorflow as tf
from datetime import datetime

class Singleton:
    def __init__(self):

        date_obj = datetime.now()
        date_str = date_obj.strftime("%Y-%m-%d-%H:%M:%S")
        self.models_path = './models/model_{}'.format(date_str)

        self.value = "I'm a singleton instance"
        self.K = 4
        self.tm2demands = []
        self.tm2demands_log = []

        self.link_loads_mlu = []
        self.link_loads_glb = []

        self.sdi2path = {}
        #保存结果

        self.opti_link_loads = []
        self.crit_link_loads = []
        self.topk_link_loads = []
        self.ecmp_link_loads = []
        self.glob_link_loads = []

        self.glob_solutions = []
        
        self.feeg_solutions = []
        self.feeg_link_loads = []


        self.num_epoch = 5000
        self.model_save = True
        

    def some_method(self):
        print(self.value)

    

singleton = Singleton()


def calualte_solution_load(sol,tm_idx):
    loads = np.zeros(singleton.num_links)

    for idx,o in enumerate(sol):
        s,d = singleton.idx2sd[idx]
        for i,oi in enumerate(o):
            for e in singleton.sdi2edge[(s,d,i)]:
                loads[singleton.edge2idx[e]] += oi*singleton.traffic_matrices[tm_idx][s][d]

    ll = [load/singleton.link_capacities[i] for i,load in enumerate(loads)]
    return np.array(ll)


def shuffled_fun(solution):
    for i,line in enumerate(solution):
        index_list = singleton.rand[i]
        shuffled_list = [line[i] for i in index_list]
        solution[i] = shuffled_list

def restored_fun(solution):
    for i,line in enumerate(solution):
        index_list = singleton.rand[i]
        restored_list = np.zeros((singleton.K))
        for new_idx, old_idx in enumerate(index_list):
            restored_list[old_idx] = line[new_idx]
        solution[i] = restored_list


def add_fun(solution):
    for idx,line in enumerate(solution):
        for i in reversed(range(0, singleton.K)):
            if singleton.sdi2path[idx][i]!=i:
                d = singleton.sdi2path[idx][i]
                line[d] += line[i]
                line[i] = 0

# 创建LearningRateScheduler回调函数实例，传入学习率调整函数
def lr_schedule(epoch):
    initial_lr = 0.0008
    final_lr = 0.0001
    decay_per_epoch = (initial_lr - final_lr) / singleton.num_epoch  # 假设训练100个epochs，计算每个epoch的学习率下降量
    return max(0, initial_lr - epoch * decay_per_epoch)  # 确保学习率不小于0



# 为kl_divergence_loss_sum函数添加get_config方法，用于指定保存时的配置信息
    

'''
value
K
glob_link_loads
glob_solutions
tm2demands
tm2demands_log
link_loads_mlu
link_loads_glb
data_dir
topology_name
topology_file
shortest_paths_file
Kshortest_paths_file
DG
num_nodes
num_links
idx2edge
edge2idx
link_capacities
link_weights
link_capacities_map
link_weights_map
idx2sd
sd2idx
shortest_paths
sdi2edge
edge2sdi
sd2edge
edge2sd
num_pairs
edge_adj
adjee
adjfe
adjef
topology
traffic_file
traffic_matrices
tm_cnt
traffic
shortest_paths_node
shortest_paths_link
'''