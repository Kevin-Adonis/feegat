

import numpy as np
import tensorflow as tf


from network import custom_objects,kl_divergence_loss_sum



m = tf.keras.models.load_model('/home/aerith/dev/feegat/models/model_2024-11-25-20:27:01/e4500_l1313.2415.h5', custom_objects=custom_objects)
    # 重新编译模型
    #m.compile(optimizer='adam', loss=kl_divergence_loss_sum)



m.summary()

