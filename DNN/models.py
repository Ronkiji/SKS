import logging
from keras import Model
import keras.backend as K
from keras.layers  import Dense, Dropout, Embedding, Input, concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D
from my_layers import HSMMBottom, HSMMTower, MultiHeadAttention
import tensorflow as tf
from w2vEmbReader import W2VEmbReader as EmbReader

logger = logging.getLogger(__name__)



def create_model(args, overal_maxlen, ruling_dim, vocab, num_class):
	# Create Model
	if args.model_type == 'cls':
		raise NotImplementedError


	elif args.model_type == 'HHMM_transformer':
		logger.info('Building a HHMM_transfermer')

		# Define input layers
		task_num = 2  # Task数量2个
		sequence_input_word = Input(shape=(overal_maxlen,), dtype='int32', name='sequence_input')
		taskid_input = Input(shape=(task_num,),dtype='float32',name='taskid_input')
		ruling_input = Input(shape=(ruling_dim,), dtype='float32')

		# Embedding layers
		embedded_sequences_word = Embedding(len(vocab), args.emb_dim, name='emb')(sequence_input_word)
		emb_ruling = Embedding(len(vocab), 100)(ruling_input)
		emb_output = concatenate([embedded_sequences_word, emb_ruling], axis=-1)

		#xs=Permute((3,1,2))(embedded_sequences)
		# embedded_sequences_char=Dropout(0.1)(embedded_sequences_char)

		# HSSM Layer
		tower_outputs=[]
		# task_num=2  # Task数量2个
		print('args.non_gate', args.non_gate)
		expert_outputs = HSMMBottom(args.model_type, args.non_gate, expert_units=[150, 150], gate_unit=150, task_num=task_num)(emb_output)
		# out = HSMMTower(units=[50,2])(expert_outputs)
		for i in range(task_num):
			tower_outputs.append(HSMMTower(units=[50, 2])(expert_outputs[:,i,:]))

		# Wrapper class
		class WrapperLayer(tf.keras.layers.Layer):
			def call(self, x):
				tower_outputs, taskid_input = x
				out = tf.matmul(tf.stack(tower_outputs,axis=-1),tf.expand_dims(taskid_input,-1))
				return tf.squeeze(out, axis=-1)
		
		# final prediction
		pred = WrapperLayer()([tower_outputs, taskid_input])
		# out = tf.matmul(tf.stack(tower_outputs,axis=-1),tf.expand_dims(taskid_input,-1))  ## 交替训练
		# pred = tf.squeeze(out, axis=-1)
		# pred = tf.nn.softmax(out,axis=-1)
		model = tf.keras.Model(inputs=[sequence_input_word, taskid_input, ruling_input], outputs=pred)
		model.emb_index = 0
		model.summary()

	


	elif args.model_type == 'Trm':
		logger.info("Building a Simple Word Embedding Model")

		# input
		input = Input(shape=(overal_maxlen,), dtype='int32')
		# input2 = Input(shape=(dim_ruling,300), dtype='float32')

		# embedded layer
		emb_output = Embedding(len(vocab), args.emb_dim, name='emb')(input)
		# mlp_output = Self_Attention(300)(emb_output)

		# multi-head attention layer
		mlp_output = MultiHeadAttention(300)(emb_output)
		mlp_output = Dense(300, activation='relu')(mlp_output)
		# mlp_output = Dropout(0.2)(mlp_output)

		# pooling and dense layers
		avg = GlobalAveragePooling1D()(mlp_output)
		max1 = GlobalMaxPooling1D()(mlp_output)
		concat = concatenate([avg, max1], axis=-1)
		dense1 = Dense(50, activation='relu')(concat)
		dense2 = Dense(50, activation='relu')(dense1)
		dropout = Dropout(0.5)(dense2)
		output = Dense(num_class, activation='softmax')(dropout)
		model = Model(inputs=input, outputs=output)
		model.emb_index = 1
		model.summary()



	logger.info('  Done')

	# Initialize embeddings if requested
	if args.emb_path and args.model_type not in {'FNN', 'CNN', 'HHMM'}:
		logger.info('Initializing lookup table')
		emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
		# embedding_matrix = emb_reader.get_emb_matrix_given_vocab(vocab)
		# model.layers[model.emb_index].W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.layers[model.emb_index].W.get_value()))
		model.get_layer(name='emb').set_weights(emb_reader.get_emb_matrix_given_vocab(vocab, model.get_layer(name='emb').get_weights()))  # 升级至2.0.8
		logger.info('  Done')

	return model

# Function to expand dimensions
def expand_dim(x):
	return K.expand_dims(x, 1)

# Function for matrix multiplication
def matmul(conv_output, swem_output, gate_output):
	K.dot(K.stack([conv_output, swem_output], axis=1), gate_output)

