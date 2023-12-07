import keras as keras
import keras.backend as K
# import keras.tf_keras.keras.backend as K
from keras.layers import GlobalMaxPooling1D, Dense, Dropout, GlobalAveragePooling1D, Concatenate, Layer
import tensorflow as tf


# BaseLayer classs for building layers
class BaseLayer(keras.layers.Layer):
    def build_layers(self, input_shape):
        shape = input_shape
        for layer in self.layers:
            layer.build(shape)
            shape=layer.compute_output_shape(shape)


# ExpertModule_trm class defining an expert module for the HSMM model
class ExpertModule_trm(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.conv_layers = []
        self.pooling_layers = []
        self.shapes=[]
        # self.filter_size=[5,3]
        self.filters=128
        self.layers = []
        super(ExpertModule_trm, self).__init__(**kwargs)


    def build(self, input_shape):
        # Define layers for the expert module
        self.layers.append(MultiHeadAttention(4, 100))
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(400, activation='relu'))
        self.layers.append(GlobalMaxPooling1D())
        self.layers.append(GlobalAveragePooling1D())
        self.layers.append(Concatenate())
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(self.units[0], activation='relu'))
        self.layers.append(Dense(self.units[1], activation='relu'))
        self.layers.append(Dropout(0.1))
        # self.layers.append(Add())


        super(ExpertModule_trm,self).build(input_shape)


    def call(self, inputs):
        # Connect layers in the expert module
        xs = self.layers[0](inputs)
        xs = self.layers[1](xs)
        xs = self.layers[2](xs)
        xs_max = self.layers[3](xs)
        xs_avg = self.layers[4](xs)
        xs = self.layers[5]([xs_max, xs_avg])
        for layer in self.layers[6:]:
            xs=layer(xs)
        return xs

    def compute_output_shape(self, input_shape):
        return input_shape[0]+[self.units[1]]


# GateModule class defining a gate module for the HSMM model
class GateModule(BaseLayer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.conv_layers = []
        self.pooling_layers=[]
        self.layers = []
        super(GateModule, self).__init__(**kwargs)

    def build(self, input_shape):
        # Define layers for the gate module
        self.layers.append(MultiHeadAttention(4, 100))
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(400, activation='relu'))
        self.layers.append(GlobalMaxPooling1D())
        self.layers.append(GlobalAveragePooling1D())
        self.layers.append(Concatenate())
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(self.units[0], activation='relu'))
        self.layers.append(Dense(self.units[0], activation='relu'))
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(self.units[1], activation='softmax'))


        super(GateModule,self).build(input_shape)


    def call(self, inputs):
        # Connect layers in the gate module
        xs = self.layers[0](inputs)
        xs = self.layers[1](xs)
        xs = self.layers[2](xs)
        xs_max = self.layers[3](xs)
        xs_avg = self.layers[4](xs)
        xs = self.layers[5]([xs_max, xs_avg])
        for layer in self.layers[6:]:
            xs=layer(xs)
        return xs

    def compute_output_shape(self, input_shape):
        return input_shape[0]+[self.units[-1]]


# HSMMBottom class defining the bottom layer of the HSMM model
class HSMMBottom(BaseLayer):
    # Hate Speech Mixture Model
    def __init__(self,
                 model_type,
                 non_gate,
                 expert_units,
                 gate_unit=100,
                 task_num=2, expert_num=3,
                 **kwargs):
        self.model_type = model_type
        self.non_gate = non_gate
        self.gate_unit = gate_unit
        self.expert_units = expert_units
        self.task_num = task_num
        self.expert_num = expert_num
        self.experts=[]
        self.gates=[]
        super(HSMMBottom, self).__init__(**kwargs)

    def build(self,input_shape):
         # Build expert and gate modules
        for i in range(self.expert_num):
            expert = ExpertModule_trm(units=self.expert_units)
            expert.build(input_shape)
            self.experts.append(expert)
        for i in range(self.task_num):
            gate = GateModule(units=[self.gate_unit, self.expert_num])
            gate.build(input_shape)
            self.gates.append(gate)
        super(HSMMBottom,self).build(input_shape)

    def call(self, inputs):
        # Build multiple experts
        # 构建多个expert
        expert_outputs=[]
        for expert in self.experts:
            expert_outputs.append(expert(inputs))

        # 构建多个gate，用来加权expert
        gate_outputs=[]
        if self.non_gate:
            print('1111111111111111111111111无门控')
            self.expert_output = tf.stack(expert_outputs,axis=1) # batch_size, expert_num, expert_out_dim
            m1 = tf.reduce_mean(self.expert_output, axis=1)
            outputs = tf.stack([m1, m1], axis=1)
            return outputs

        else:
            for gate in self.gates:
                gate_outputs.append(gate(inputs))
            # 使用gate对expert进行加权平均
            self.expert_output=tf.stack(expert_outputs,axis=1) # batch_size, expert_num, expert_out_dim
            self.gate_output=tf.stack(gate_outputs,axis=1) # batch_size, task_num, expert_num
            outputs=tf.matmul(self.gate_output,self.expert_output) # batch_size,task_num,expert_out_dim
            return outputs

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.task_num, self.expert_units[-1]]

# HSMMTower class defining the tower layer of the HSMM model
class HSMMTower(BaseLayer):
    # Hate Speech Mixture Model Tower
    def __init__(self,
                 units,
                 **kwargs):
        self.units = units
        self.layers=[]
        super(HSMMTower, self).__init__(**kwargs)

    def build(self, input_shape):
        # Build layers for the tower
        for unit in self.units[:-1]:
            self.layers.append(Dense(unit, activation='relu'))
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(self.units[-1], activation='softmax'))
        self.build_layers(input_shape)
        super(HSMMTower,self).build(input_shape)

    def call(self, inputs):
        # Connect layers in the tower
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.units[-1]]



class MultiHeadAttention(Layer):
    """
	多头注意力机制
	"""
    def __init__(self,heads, head_size, output_dim=None, **kwargs):
        self.heads = heads
        self.head_size = head_size
        self.output_dim = output_dim or heads * head_size
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Build the trainable weights for the Multi-Head Attention layer
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        he_initialize = keras.initializers.he_normal()
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2], self.head_size),
                                      initializer=he_initialize,
                                      trainable=True)
        self.dense = self.add_weight(name='dense',
                                     shape=(input_shape[2], self.output_dim),
                                     initializer=he_initialize,
                                     trainable=True)

        super(MultiHeadAttention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        # Perform the Multi-Head Attention computation
        out = []
        for i in range(self.heads):
            WQ = K.dot(x, self.kernel[0])
            WK = K.dot(x, self.kernel[1])
            WV = K.dot(x, self.kernel[2])

            # print("WQ.shape",WQ.shape)
            # print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)

            QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
            QK = QK / (100**0.5)
            QK = K.softmax(QK)

            # print("QK.shape",QK.shape)

            V = K.batch_dot(QK,WV)
            out.append(V)
        out = Concatenate(axis=-1)(out)
        # Project the concatenated output using a dense layer
        out = K.dot(out, self.dense)
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)


