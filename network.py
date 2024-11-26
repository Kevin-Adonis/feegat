from __future__ import absolute_import
from top import singleton
import numpy as np


import tensorflow as tf

import os
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dropout, LeakyReLU
from tensorflow.keras.activations import softmax

from datetime import datetime
from tensorflow.keras.callbacks import Callback


class LastBipartiteGraphMultiHeadAttentionLayer(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 activation='relu',
                 kernels_f = None,
                 kernels_e = None,
                 kernels_a = None,
                 **kwargs):


        self.Hdim = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads
        self.output_dim = self.Hdim
        self.attn_heads_reduction = attn_heads_reduction
        self.activation = activations.get(activation)

        self.kernels_f = []
        self.kernels_e = []
        self.kernels_a = []

        if kernels_f is not None:
            self.kernels_f = kernels_f
        if kernels_e is not None:
            self.kernels_e = kernels_e
        if kernels_a is not None:
            self.kernels_a = kernels_a
        
        super(LastBipartiteGraphMultiHeadAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        Fdim = input_shape[0][-1]
        Edim = input_shape[1][-1]

        for head in range(self.attn_heads):

            kernel_f = self.add_weight(shape=(Fdim, self.Hdim),
                                     initializer='glorot_uniform',
                                     regularizer=None,
                                     constraint=None,
                                     name='kernel_f2e_{}'.format(head))
            

            kernel_e = self.add_weight(shape=(Edim, self.Hdim),
                                     initializer='glorot_uniform',
                                     regularizer=None,
                                     constraint=None,
                                     name='kernel_e2f_{}'.format(head))
            

            attn_kernel_f2e = self.add_weight(shape=(self.Hdim, 1),
                                               initializer='glorot_uniform',
                                               regularizer=None,
                                               constraint=None,
                                               name='attn_kernel_f2e_{}'.format(head))
            
            self.kernels_f.append(kernel_f)
            self.kernels_e.append(kernel_e)
            self.kernels_a.append(attn_kernel_f2e)


        self.built = True
        super(LastBipartiteGraphMultiHeadAttentionLayer, self).build(input_shape)

    def _compute_attention(self, h_src, h_dst, A, attn_kernel):
        # 计算注意力系数
        src_expanded = K.expand_dims(h_src, axis=2)  # (num_nodes_src, 1, output_dim)
        dst_expanded = K.expand_dims(h_dst, axis=1)  # (1, num_nodes_dst, output_dim)
        # 计算 e_{ij}
        e_src = K.dot(src_expanded, attn_kernel)  # 形状为 (num_nodes_src, 1, 1)
        e_dst = K.dot(dst_expanded, attn_kernel)

        e = tf.add(e_src, e_dst)
        e = tf.squeeze(e, axis=-1)
        e = tf.nn.leaky_relu(e)
        e = tf.where(A > 0, e, -1e9)
        # 对邻居节点的注意力系数进行 softmax 归一化
        alpha = softmax(e, axis=-1)

        # 对邻居节点的特征进行加权求和
        h_prime = tf.matmul(alpha, h_dst)  # (num_nodes_src, output_dim)
        return h_prime

    def call(self, inputs):

        F0,E0 = inputs
        attn_scores_fe = []

        for head in range(self.attn_heads):
            We = self.kernels_e[head]
            Wf = self.kernels_f[head]
            E1 = K.dot(E0,We)
            F1 = K.dot(F0,Wf)
            attn_fe = self._compute_attention(F1, E1, singleton.adjfe, self.kernels_a[head])
            attn_scores_fe.append(attn_fe)
            

        output_fe = K.mean(K.stack(attn_scores_fe), axis=0)
        #output_fe = self.activation(output_fe) 
        output_fe = softmax(output_fe, axis=-1)

        return output_fe

    def compute_output_shape(self, input_shape):
         return (input_shape[0][1], self.output_dim)

    def get_config(self):
        config = super(LastBipartiteGraphMultiHeadAttentionLayer, self).get_config()
        config.update({
            'F_': self.Hdim,
            'attn_heads': self.attn_heads,
            'attn_heads_reduction': self.attn_heads_reduction,
            'activation': activations.serialize(self.activation) if self.activation is not None else None,
            'kernels_f': [kernel.numpy().tolist() for kernel in self.kernels_f],
            'kernels_e': [kernel.numpy().tolist() for kernel in self.kernels_e],
            'kernels_a': [kernel.numpy().tolist() for kernel in self.kernels_a],
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            F_=config['F_'],
            attn_heads=config['attn_heads'],
            attn_heads_reduction=config['attn_heads_reduction'],
            activation=activations.deserialize(config['activation']),
            kernels_f=[np.array(kernel) for kernel in config['kernels_f']],
            kernels_e=[np.array(kernel) for kernel in config['kernels_e']],
            kernels_a=[np.array(kernel) for kernel in config['kernels_a']],
        )



class BipartiteGraphMultiHeadAttentionLayer(Layer):

    def __init__(self,
                 F_,
                 attn_heads=1,
                 attn_heads_reduction='concat',  # {'concat', 'average'}
                 activation='relu',
                 kernels_f = None,
                 kernels_e = None,
                 kernels_a = None,
                 **kwargs):
        if attn_heads_reduction not in {'concat', 'average'}:
            raise ValueError('Possbile reduction methods: concat, average')

        self.Hdim = F_  # Number of output features (F' in the paper)
        self.attn_heads = attn_heads  # Number of attention heads (K in the paper)
        self.attn_heads_reduction = attn_heads_reduction  # Eq. 5 and 6 in the paper
        self.activation = activations.get(activation)  # Eq. 4 in the paper
    
        if attn_heads_reduction == 'concat':
            # Output will have shape (..., K * F')
            self.output_dim = self.Hdim * self.attn_heads
        else:
            # Output will have shape (..., F')
            self.output_dim = self.Hdim


        self.kernels_f = []
        self.kernels_e = []
        self.kernels_a = []

        if kernels_f is not None:
            self.kernels_f = kernels_f
        if kernels_e is not None:
            self.kernels_e = kernels_e
        if kernels_a is not None:
            self.kernels_a = kernels_a

        super(BipartiteGraphMultiHeadAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        Fdim = input_shape[0][-1]
        Edim = input_shape[1][-1]

        for head in range(self.attn_heads):

            kernel_f = self.add_weight(shape=(Fdim, self.Hdim),
                                     initializer='glorot_uniform',
                                     regularizer=None,
                                     constraint=None,
                                     name='kernel_f2e_{}'.format(head))
            
            kernel_e = self.add_weight(shape=(Edim, self.Hdim),
                                     initializer='glorot_uniform',
                                     regularizer=None,
                                     constraint=None,
                                     name='kernel_e2f_{}'.format(head))
        
            attn_kernel_f2e = self.add_weight(shape=(self.Hdim, 1),
                                               initializer='glorot_uniform',
                                               regularizer=None,
                                               constraint=None,
                                               name='attn_kernel_f2e_{}'.format(head),)
            attn_kernel_e2f = self.add_weight(shape=(self.Hdim, 1),
                                                 initializer='glorot_uniform',
                                                 regularizer=None,
                                                 constraint=None,
                                                 name='attn_kernel_e2f_{}'.format(head))
            
            self.kernels_f.append(kernel_f)
            self.kernels_e.append(kernel_e)
            self.kernels_a.append([attn_kernel_f2e, attn_kernel_e2f])

            
        self.dp = Dropout(0.5)

        self.built = True
        super(BipartiteGraphMultiHeadAttentionLayer, self).build(input_shape)

    def _compute_attention(self, h_src, h_dst, A, attn_kernel):
        # 计算注意力系数
        src_expanded = K.expand_dims(h_src, axis=2)  # (batch_size,num_nodes_src, 1, output_dim)
        dst_expanded = K.expand_dims(h_dst, axis=1)  # (batch_size,1, num_nodes_dst, output_dim)
        # 计算 e_{ij}

        e_src = K.dot(src_expanded, attn_kernel)  # 形状为 (batch_size,num_nodes_src, 1, 1)
        e_dst = K.dot(dst_expanded, attn_kernel)

        e = tf.add(e_src, e_dst)
        e = tf.squeeze(e, axis=-1)
        e = tf.nn.leaky_relu(e)
        e = tf.where(A > 0, e, -1e9)
        alpha = softmax(e, axis=-1)

        #alpha = self.drop_out1(alpha)
        h_dst = self.dp(h_dst)

        h_prime = tf.matmul(alpha, h_dst)  # (num_nodes_src, output_dim)
        return h_prime

    def call(self, inputs):
        # F0 流特征
        # E0 边特征
        # Adjfe 流对边的adj
        # Adjef 边对流的adj（是上面的转置矩阵？）

        F0,E0 = inputs

        attn_scores_fe = []
        attn_scores_ef = []

        for head in range(self.attn_heads):
            We = self.kernels_e[head]
            Wf = self.kernels_f[head]
            E1 = tf.matmul(E0,We)
            F1 = tf.matmul(F0,Wf)

            attn_fe = self._compute_attention(F1, E1, singleton.adjfe, self.kernels_a[head][0])
            # attn_fe = K.bias_add(attn_fe, self.biases_f[head])
            attn_scores_fe.append(attn_fe)

            attn_ef = self._compute_attention(E1, F1, singleton.adjef, self.kernels_a[head][1])
            # attn_ef = K.bias_add(attn_ef, self.biases_e[head])
            attn_scores_ef.append(attn_ef)

        # Aggregate the heads' output according to the reduction method
        if self.attn_heads_reduction == 'concat':
            output_fe = K.concatenate(attn_scores_fe, axis=-1) # 形状为 (num_nodes_f, output_dim * num_heads)
            output_ef = K.concatenate(attn_scores_ef, axis=-1) # 形状为 (num_nodes_e, output_dim * num_heads)
        else:
            output_fe = K.mean(K.stack(attn_scores_fe), axis=0)
            output_ef = K.mean(K.stack(attn_scores_ef), axis=0)   # N x F')


        output_fe = softmax(output_fe, axis=-1)
        #output_fe = self.activation(output_fe) 
        output_ef = self.activation(output_ef) 
        
        return output_fe, output_ef

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][1], self.output_dim),
                (input_shape[1][1], self.output_dim)]

    def get_config(self):
        config = super(BipartiteGraphMultiHeadAttentionLayer, self).get_config()
        config.update({
            'F_': self.Hdim,
            'attn_heads': self.attn_heads,
            'attn_heads_reduction': self.attn_heads_reduction,
            'activation': activations.serialize(self.activation) if self.activation is not None else None,
            'kernels_f': [kernel.numpy().tolist() for kernel in self.kernels_f],
            'kernels_e': [kernel.numpy().tolist() for kernel in self.kernels_e],
            'kernels_a': [[inner_kernel.numpy().tolist() for inner_kernel in pair] for pair in self.kernels_a],
        })
        return config

    @classmethod
    def from_config(cls, config):
        instance = cls(
            F_=config['F_'],
            attn_heads=config['attn_heads'],
            attn_heads_reduction=config['attn_heads_reduction'],
            activation=activations.deserialize(config['activation'])
        )

        instance.kernels_f = [np.array(kernel) for kernel in config['kernels_f']]
        instance.kernels_e = [np.array(kernel) for kernel in config['kernels_e']]
        instance.kernels_a = []
        for pair in config['kernels_a']:
            reconstructed_pair = [np.array(inner_kernel) for inner_kernel in pair]
            instance.kernels_a.append(reconstructed_pair)

        return instance


class GraphMultiHeadAttentionLayer(Layer):
    def __init__(self, output_dim, num_heads=1,kernels=None,attn_kernels = None, **kwargs):
        """
        普通图的多头GAT层，实现图中每个节点与其邻居节点的多头注意力聚合。
        参数:
            output_dim: 每个注意力头的输出维度
            num_heads: 注意力头的数量
        """
        self.output_dim = output_dim
        self.num_heads = num_heads

        self.kernels = []
        self.attn_kernels = []


        if kernels is not None:
            self.kernels = kernels
        if attn_kernels is not None:
            self.attn_kernels = attn_kernels

        self.dp = Dropout(0.5)

        super(GraphMultiHeadAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]  # 节点特征的输入维度
        
        # 定义每个注意力头的线性变换权重矩阵
        self.kernels = [self.add_weight(shape=(input_dim, self.output_dim),
                                        initializer='glorot_uniform',
                                        name=f'kernel_{i}') for i in range(self.num_heads)]
        
        # self.biases = [self.add_weight(shape=(self.output_dim,),
        #                                initializer='zeros',
        #                                name=f'bias_{i}') for i in range(self.num_heads)]
        
        # 定义每个注意力头的注意力权重向量
        self.attn_kernels = [self.add_weight(shape=(self.output_dim, 1),
                                             initializer='glorot_uniform',
                                             name=f'attn_kernel_{i}') for i in range(self.num_heads)]

        self.built = True
        super(GraphMultiHeadAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        X = inputs[0]  # X是节点特征，A是邻接矩阵

        # 针对每个注意力头计算特征变换和注意力分数
        attn_heads = []
        for i in range(self.num_heads):
            h_i = K.dot(X, self.kernels[i])  # 形状为 (batch_size, num_nodes, output_dim)
            attn = self._compute_attention(h_i, singleton.adjee, self.attn_kernels[i])
            # attn = K.bias_add(attn, self.biases[i])
            attn_heads.append(attn)  # 形状为 (batch_size, num_nodes, output_dim)

        # 将所有注意力头的输出拼接在一起
        #output = K.concatenate(attn_heads, axis=-1)  # 形状为 (batch_size, num_nodes, output_dim * num_heads)
        output = K.mean(K.stack(attn_heads), axis=0)
        output = tf.nn.leaky_relu(output)
        
        return output

    def _compute_attention(self, h, A, attn_kernel):

        h_expanded_i = K.expand_dims(h, axis=2)  # 形状为 (batch_size, num_nodes, 1, output_dim)
        h_expanded_j = K.expand_dims(h, axis=1)  # 形状为 (batch_size, 1, num_nodes, output_dim)
        e_i = K.dot(h_expanded_i, attn_kernel)  # 形状为 (batch_size, num_nodes, 1, 1)
        e_j = K.dot(h_expanded_j, attn_kernel)  # 形状为 (batch_size, 1, num_nodes, 1)

        e = tf.add(e_i, e_j)  # 形状为 (batch_size, num_nodes, num_nodes, 1)
        e = K.squeeze(e, axis=-1)  # 去掉多余的维度，形状为 (batch_size, num_nodes, num_nodes)
        e = K.relu(e)
        e = tf.where(A > 0, e, -1e9)  # 形状为 (batch_size, num_nodes, num_nodes)

        alpha = softmax(e, axis=-1)  # 形状为 (batch_size, num_nodes, num_nodes)

        #alpha = self.drop_out1(alpha)
        h = self.dp(h)

        h_prime = K.batch_dot(alpha, h)  # 形状为 (batch_size, num_nodes, output_dim)

        return h_prime

    def compute_output_shape(self, input_shape):
        return (input_shape[0][1], self.output_dim)

    def get_config(self):
        config = super(GraphMultiHeadAttentionLayer, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'num_heads': self.num_heads,
            'kernels': [kernel.numpy().tolist() for kernel in self.kernels],
            'attn_kernels': [kernel.numpy().tolist() for kernel in self.attn_kernels],
        })
        return config

    @classmethod
    def from_config(cls, config):
        instance = cls(
            output_dim=config['output_dim'],
            num_heads=config['num_heads']
        )
        instance.kernels = [np.array(kernel) for kernel in config['kernels']]
        instance.attn_kernels = [np.array(kernel) for kernel in config['attn_kernels']]
        return instance




class LossHistory(Callback):
    def __init__(self):
        super(LossHistory, self).__init__()
        self.best_loss = np.inf  # 初始化为正无穷，确保第一次的损失值能被更新
        self.history = []
        self.init = False

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.history.append(logs.copy())

        if singleton.model_save:
            if epoch > singleton.num_epoch/5 and epoch % 100 == 0:
                current_loss = logs.get('loss')  # 获取当前 epoch 的损失值
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    formatted_loss = "{:.4f}".format(current_loss)

                    self.model.save(f'{singleton.models_path}/e{epoch}_l{formatted_loss}.h5')  # 保存模型，这里文件名可根据需要修改
    
    def on_train_end(self,logs):
        self.model.save(f'{singleton.models_path}/final_model.h5')

        import json
        with open(f'{singleton.models_path}/history.json', 'w') as f:
            json.dump(self.history, f)


def getLayer(heads,dims):
    return BipartiteGraphMultiHeadAttentionLayer(dims,heads,'average'),GraphMultiHeadAttentionLayer(dims,heads)

class FEEGAT(tf.keras.Model):
    def __init__(self):
        super(FEEGAT, self).__init__()

        self.callback = LossHistory()
        self.num_heads = 4       # 注意力头的数量
        self.set = singleton.set

        self.mylayers = []
        
    def build(self, input_shape):
        super(FEEGAT, self).build(input_shape)

        num_heads = self.num_heads

        for i in self.set:
            layer0,layer1 = getLayer(num_heads,i)
            self.mylayers.append([layer0,layer1])

        self.layer_last = LastBipartiteGraphMultiHeadAttentionLayer(singleton.K,num_heads,'average')


    def call(self, inputs):

        f, e = inputs

        for layers in self.mylayers:
            l1 = layers[0]
            l2 = layers[1]
            f,e = l1([f, e])
            e = l2([e])

        f = self.layer_last([f, e])

        return f
    def get_config(self):
        config = super(FEEGAT, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'mylayers': [],
        })
        for layer_pair in self.mylayers:
            layer0_config = tf.keras.layers.serialize(layer_pair[0])
            layer1_config = tf.keras.layers.serialize(layer_pair[1])
            config['mylayers'].append([layer0_config, layer1_config])
        layer_last_config = tf.keras.layers.serialize(self.layer_last)
        config['layer_last_config'] = layer_last_config
        return config
    
    @classmethod
    def from_config(cls, config):
        instance = cls()
        instance.num_heads = config['num_heads']
        instance.mylayers = []
        for layer_pair_config in config['mylayers']:
            layer0 = tf.keras.layers.deserialize(layer_pair_config[0])
            layer1 = tf.keras.layers.deserialize(layer_pair_config[1])
            instance.mylayers.append([layer0, layer1])
        instance.layer_last = tf.keras.layers.deserialize(config['layer_last_config'])
        return instance

   
    
#@tf.function
def kl_divergence_loss_sum(y_true, y_pred,epsilon=1e-10):
    # 计算每一行的 KL 散度
    batch_size = tf.shape(y_true)[0]

    y_true = tf.clip_by_value(y_true, epsilon, 1.0)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)

    print('y_pred.shape',y_pred.shape)
    print('y_true.shape',y_true.shape)

    row_kl_divergences = tf.map_fn(
        lambda x: tf.reduce_sum(x[0] * tf.math.log(x[0] / x[1]), axis=-1),
        (y_true, y_pred),
        fn_output_signature=tf.float32
    )
    # 计算所有行的 KL 散度总和
    #total_kl_divergence = tf.reduce_sum(row_kl_divergences)/tf.cast(batch_size, tf.float32)
    total_kl_divergence = tf.reduce_sum(row_kl_divergences)
    return total_kl_divergence


custom_objects = {
    'kl_divergence_loss_sum': kl_divergence_loss_sum,
    'BipartiteGraphMultiHeadAttentionLayer': BipartiteGraphMultiHeadAttentionLayer,
    'GraphMultiHeadAttentionLayer': GraphMultiHeadAttentionLayer,
    'LastBipartiteGraphMultiHeadAttentionLayer': LastBipartiteGraphMultiHeadAttentionLayer,
    'FEEGAT': FEEGAT
}

