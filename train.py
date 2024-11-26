from __future__ import print_function

import numpy as np
from absl import app
from absl import flags

from env import Environment
from game import CFRRL_Game
from config import get_config

from top import singleton
from top import shuffled_fun,lr_schedule,restored_fun,add_fun,calualte_solution_load

import pickle
import json
from networkx.readwrite import json_graph

from network import FEEGAT,kl_divergence_loss_sum
from network import custom_objects

import tensorflow as tf
from draw import draw_mlu,draw_sum

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')
flags.DEFINE_boolean('eval_delay', False, 'evaluate delay or not')

def main(_):
    np.set_printoptions(suppress=True)
    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=False)
    game = CFRRL_Game(config, env)

    tm_cnt = singleton.tm_cnt

    singleton.glob_solutions = np.load('./res/glob_solutions.npy')
    singleton.glob_link_loads = np.load('./res/glob_link_loads.npy')
    singleton.opti_link_loads = np.load('./res/opti_link_loads.npy')
    singleton.topk_link_loads = np.load('./res/topk_link_loads.npy')
    singleton.crit_link_loads = np.load('./res/crit_link_loads.npy')
    singleton.ecmp_link_loads = np.load('./res/ecmp_link_loads.npy')

    for solution in singleton.glob_solutions[:]:
        add_fun(solution)
        #shuffled_fun(solution)

    print('============================================================')


    t1 = singleton.glob_solutions[:tm_cnt-1]
    t2 = np.array(singleton.tm2demands_log[1:tm_cnt]).reshape(len(t1),132,1)
    #t2 = np.array(singleton.tm2demands[1:tm_cnt]).reshape(len(t1),132,1)
    x1 = np.concatenate((t1,t2),axis=-1)
    x2 = np.expand_dims(singleton.glob_link_loads[:tm_cnt-1], axis=-1)
    x = [x1,x2]

    y = np.array(singleton.glob_solutions[1:tm_cnt])

    print('x1.shape:',np.shape(x1))
    print('x2.shape:',np.shape(x2))
    print(' y.shape:',np.shape(y)) 

    num_epoch = singleton.num_epoch
    batch_size = singleton.batch_size

    model = FEEGAT()

  
    sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)
    rms_prop = tf.keras.optimizers.RMSprop(learning_rate=0.0005)
    adam = tf.keras.optimizers.Adam()

    model.compile(optimizer=adam, loss=kl_divergence_loss_sum)

    lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    model.fit(x,y,epochs=num_epoch,batch_size=batch_size,callbacks = [model.callback,lr_scheduler_callback])
    #model.fit(x,y,epochs=singleton.num_epoch,batch_size=16,callbacks = [model.callback])
    model.summary()

    model.save('test_model.h5',overwrite = True)

    m = tf.keras.models.load_model('test_model.h5', custom_objects=custom_objects)
    m.summary()

    predictions = model.predict(x)
    np.save(f'{singleton.models_path}/predictions.npy', np.array(predictions))

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


    for i,pre in enumerate(predictions[:20]):
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
        'ecmp_mlu': ecmp_sum,
        'feeg_mlu': feeg_sum
    }

    draw_mlu(data_mlu)
    draw_sum(data_sum)




    # for k in singleton.__dict__.keys():
    #     print(k)
   
    return


if __name__ == '__main__':
    app.run(main)
