

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
from draw import draw_mlu,draw_sum,draw_box,draw_cdf


from top import singleton


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

    predictions = np.load('/home/aerith/dev/feegat/models/model_2024-11-26-23:19:52/predictions.npy')

    for pre in predictions:
        #restored_fun(pre)
        add_fun(pre)


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


    data_mlu = {
        'glob_mlu': glob_mlu,
        'opti_mlu': opti_mlu,
        'topk_mlu': topk_mlu,
        'crit_mlu': crit_mlu,
        'ecmp_mlu': ecmp_mlu,
        'feeg_mlu': feeg_mlu
    }

    data_sum = {
        'glob_sum': glob_sum,
        'opti_sum': opti_sum,
        'topk_sum': topk_sum,
        'crit_sum': crit_sum,
        'ecmp_sum': ecmp_sum,
        'feeg_sum': feeg_sum
    }

    draw_sum(data_sum,True)
    draw_mlu(data_mlu,True)
    draw_box(data_mlu,True)
    draw_cdf(data_mlu,True)

 
main()