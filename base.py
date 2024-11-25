import os
import numpy as np
from absl import app
from absl import flags

from env import Environment
from game import CFRRL_Game
from config import get_config

from top import singleton

FLAGS = flags.FLAGS
flags.DEFINE_string('ckpt', '', 'apply a specific checkpoint')
flags.DEFINE_boolean('eval_delay', False, 'evaluate delay or not')


def sim(config, game):
    for tm_idx in game.tm_indexes[:]:
        actions = []
        game.evaluate(tm_idx, actions, eval_delay=FLAGS.eval_delay) 

def main(_):
    np.set_printoptions(suppress=True)
    config = get_config(FLAGS) or FLAGS
    env = Environment(config, is_training=False)
    game = CFRRL_Game(config, env)

    sim(config,game)

    np.save('./res/opti_link_loads.npy', np.array(singleton.opti_link_loads))
    np.save('./res/crit_link_loads.npy', np.array(singleton.crit_link_loads))
    np.save('./res/topk_link_loads.npy', np.array(singleton.topk_link_loads))
    np.save('./res/glob_link_loads.npy', np.array(singleton.glob_link_loads))
    np.save('./res/ecmp_link_loads.npy', np.array(singleton.ecmp_link_loads))
    np.save('./res/glob_solutions.npy', np.array(singleton.glob_solutions))

    singleton.glob_solutions = np.load('./res/glob_solutions.npy')
    singleton.glob_link_loads = np.load('./res/glob_link_loads.npy')
    singleton.opti_link_loads = np.load('./res/opti_link_loads.npy')
    singleton.topk_link_loads = np.load('./res/topk_link_loads.npy')
    singleton.crit_link_loads = np.load('./res/crit_link_loads.npy')
    singleton.ecmp_link_loads = np.load('./res/ecmp_link_loads.npy')

    return


if __name__ == '__main__':
    app.run(main)