

import numpy as np
from absl import app
from absl import flags
import tensorflow as tf
from env import Environment
from game import CFRRL_Game
from config import get_config
from top import shuffled_fun,lr_schedule,restored_fun,add_fun,calualte_solution_load

import pickle
import json
from networkx.readwrite import json_graph

from network import FEEGAT,kl_divergence_loss_sum
from network import custom_objects,triangular_clr,exp_range_clr

import tensorflow as tf
from draw import draw_line_mlu,draw_line_sum,draw_box_mlu,draw_cdf_cdf,draw_heatmap,draw_2dheatmap
from draw import draw_box_sum,draw_cdf_sum

from top import singleton
import matplotlib.pyplot as plt
import matplotlib


from network import custom_objects,kl_divergence_loss_sum

FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')
flags.DEFINE_boolean('eval_delay', False, 'evaluate delay or not')

def main():
    np.set_printoptions(suppress=True)
    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=False)
    game = CFRRL_Game(config, env)

    singleton.glob_solutions = np.load('./res/glob_solutions.npy')
    singleton.glob_link_loads = np.load('./res/glob_link_loads.npy')
    singleton.opti_link_loads = np.load('./res/opti_link_loads.npy')
    singleton.topk_link_loads = np.load('./res/topk_link_loads.npy')
    singleton.crit_link_loads = np.load('./res/crit_link_loads.npy')
    singleton.ecmp_link_loads = np.load('./res/ecmp_link_loads.npy')

    predictions = np.load('/home/aerith/dev/feegat/models/model_2024-11-27-01:17:26/predictions.npy')

    for pre in predictions:
        #restored_fun(pre)
        add_fun(pre)

    for solution in singleton.glob_solutions:
        add_fun(solution)



    glob_mlu = []
    opti_mlu = []
    topk_mlu = []
    crit_mlu = []
    ecmp_mlu = []
    feeg_mlu = []

    glob_sum = []
    opti_sum = []
    topk_sum = []
    crit_sum = []
    ecmp_sum = []
    feeg_sum = []

    for i,pre in enumerate(predictions[:]):

        if i < 1000 or i > 1100:
            continue

        idx = i+1

        ll = calualte_solution_load(pre,idx)
        feeg_mlu.append(np.max(ll))
        feeg_sum.append(np.sum(ll))

        # print('===========================================')
        # print('feeg: ',np.max(ll),np.sum(ll))
        # print('glob: ',np.max(singleton.glob_link_loads[idx]),np.sum(singleton.glob_link_loads[idx]))
        # print('opti: ',np.max(singleton.opti_link_loads[idx]),np.sum(singleton.opti_link_loads[idx]))
        # print('topk: ',np.max(singleton.topk_link_loads[idx]),np.sum(singleton.topk_link_loads[idx]))
        # print('ecmp: ',np.max(singleton.ecmp_link_loads[idx]),np.sum(singleton.ecmp_link_loads[idx]))
        # print('crit: ',np.max(singleton.crit_link_loads[idx]),np.sum(singleton.crit_link_loads[idx]))

        glob_mlu.append(np.max(singleton.glob_link_loads[idx]))
        glob_sum.append(np.sum(singleton.glob_link_loads[idx]))

        opti_mlu.append(np.max(singleton.opti_link_loads[idx]))
        opti_sum.append(np.sum(singleton.opti_link_loads[idx]))

        topk_mlu.append(np.max(singleton.topk_link_loads[idx]))
        topk_sum.append(np.sum(singleton.topk_link_loads[idx]))

        crit_mlu.append(np.max(singleton.crit_link_loads[idx]))
        crit_sum.append(np.sum(singleton.crit_link_loads[idx]))

        ecmp_mlu.append(np.max(singleton.ecmp_link_loads[idx]))
        ecmp_sum.append(np.sum(singleton.ecmp_link_loads[idx]))

        if i == 1050:
            tmp = pre.copy()
            tmp = np.where(tmp < 0.01, 0, tmp)
            tmp = np.where(tmp > 0.99, 1, tmp)
            tmp = np.around(tmp, decimals=6)
            vmax = max(np.max(ll),np.max(singleton.opti_link_loads[idx]))

            draw_heatmap('feeg',ll,False,vmax)
            draw_heatmap('glob',singleton.glob_link_loads[idx],False,vmax)
            draw_heatmap('opti',singleton.opti_link_loads[idx],False,vmax)
            draw_heatmap('topk',singleton.topk_link_loads[idx],False,vmax)
            draw_heatmap('ecmp',singleton.crit_link_loads[idx],False,vmax)
            draw_heatmap('crit',singleton.ecmp_link_loads[idx],False,vmax)

            
            np.set_printoptions(formatter={'float': lambda x: f"{x:0.6f}"})
            print(tmp)
            draw_2dheatmap('solution_feeg',tmp,b_show=True)
            print(np.array(singleton.glob_solutions[idx]))
            draw_2dheatmap('solution_glob',singleton.glob_solutions[idx],b_show=True)


    data_mlu = {
        'feeg_mlu': feeg_mlu,
        'glob_mlu': glob_mlu,
        'opti_mlu': opti_mlu,
        'topk_mlu': topk_mlu,
        'crit_mlu': crit_mlu,
        'ecmp_mlu': ecmp_mlu,
        
    }

    data_sum = {
        'feeg_sum': feeg_sum,
        'glob_sum': glob_sum,
        'opti_sum': opti_sum,
        'topk_sum': topk_sum,
        'crit_sum': crit_sum,
        'ecmp_sum': ecmp_sum,
        
    }

    # draw_sum(data_sum,True)
    # draw_mlu(data_mlu,True)
    # draw_box(data_mlu,True)
    # draw_cdf(data_mlu,True)

    draw_line_sum(data_sum)
    draw_box_sum(data_sum)
    draw_cdf_sum(data_sum)

    draw_line_mlu(data_mlu)
    draw_box_mlu(data_mlu)
    draw_cdf_cdf(data_mlu)
    

 
main()