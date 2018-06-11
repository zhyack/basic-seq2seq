import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.layers import core as layers_core
import math
import copy
import data_utils
import model_utils
from contrib_rnn_cell import ExtendedMultiRNNCell
from GNMTCell import GNMTAttentionMultiCell
from reward import *
import rlloss

class Seq2SeqModel():
    def __init__(self, config):

        print('The model is built for training:', config['IS_TRAIN'])

        self.rl_enable = config['RL_ENABLE']
        self.bleu_enable = config['BLEU_RL_ENABLE']

        self.learning_rate = tf.Variable(config['LR'], dtype=tf.float32, name='model_learning_rate', trainable=False)

        self.word_embedding_learning_rate = tf.Variable(config['WE_LR'], dtype=tf.float32, name='model_learning_rate', trainable=False)
        self.encoder_learning_rate = tf.Variable(config['ENCODER_LR'], dtype=tf.float32, name='model_learning_rate', trainable=False)
        self.decoder_learning_rate = tf.Variable(config['DECODER_LR'], dtype=tf.float32, name='model_learning_rate', trainable=False)
        if config['SPLIT_LR']:
            def tmp_func():
                self.word_embedding_learning_rate.assign( self.word_embedding_learning_rate * config['LR_DECAY'])
                self.encoder_learning_rate.assign( self.encoder_learning_rate * config['LR_DECAY'])
                self.decoder_learning_rate.assign( self.decoder_learning_rate * config['LR_DECAY'])
            self.lr_decay_op = tmp_func()
        else:
            self.lr_decay_op = self.learning_rate.assign(self.learning_rate * config['LR_DECAY'])

        if config['OPTIMIZER']=='Adam':
            self.optimizer = tf.train.AdamOptimizer
        elif config['OPTIMIZER']=='GD':
            self.optimizer = tf.train.GradientDescentOptimizer
        else:
            raise Exception("Wrong optimizer name...")

        self.global_step = tf.Variable(config['GLOBAL_STEP'], dtype=tf.int32, name='model_global_step', trainable=False)
        self.batch_size = config['BATCH_SIZE']
        self.input_size = config['INPUT_VOCAB_SIZE']
        self.output_size = config['OUTPUT_VOCAB_SIZE']
        self.encoder_hidden_size = config['ENCODER_HIDDEN_SIZE']
        self.decoder_hidden_size = config['DECODER_HIDDEN_SIZE']
        self.embedding_size = config['WORD_EMBEDDING_SIZE']



        self.encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, self.batch_size), name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, ), name='encoder_inputs_length')
        self.encoder_inputs_mask = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None), name='encoder_inputs_mask')
        self.decoder_inputs = tf.placeholder(dtype=tf.int32, shape=(None, self.batch_size), name='decoder_inputs')
        self.decoder_inputs_length = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, ), name='decoder_inputs_length')
        self.decoder_inputs_mask = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None), name='decoder_inputs_mask')
        self.decoder_targets = tf.placeholder(dtype=tf.int32, shape=(None, self.batch_size), name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(dtype=tf.int32, shape=(self.batch_size, ), name='decoder_targets_length')
        self.decoder_targets_mask = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None), name='decoder_targets_mask')
        self.rewards = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, None, self.output_size), name='decoder_targets_mask')


        with tf.variable_scope("WordEmbedding") as scope:
            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.input_word_embedding_matrix = tf.get_variable(name="input_word_embedding_matrix", shape=[self.input_size, self.embedding_size], initializer=initializer, dtype=tf.float32)
            self.output_word_embedding_matrix = tf.get_variable(name="output_word_embedding_matrix", shape=[self.output_size, self.embedding_size], initializer=initializer, dtype=tf.float32)

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.input_word_embedding_matrix, self.encoder_inputs)

            self.decoder_inputs_embedded = tf.nn.embedding_lookup(self.output_word_embedding_matrix, self.decoder_inputs)
            print('Embedding Trainable Variables')
            self.embedding_variables = scope.trainable_variables()
            print(self.embedding_variables)





        self.encoder_cell = []
        for _ in range(config['ENCODER_LAYERS']):
            config_encoder_cell = None
            if config['CELL'] in ['LSTM', 'lstm']:
                config_encoder_cell = tf.contrib.rnn.BasicLSTMCell(self.encoder_hidden_size)
            elif config['CELL'] in ['GRU', 'gru']:
                config_encoder_cell = tf.contrib.rnn.GRUCell(self.encoder_hidden_size)
            else:
                config_encoder_cell = tf.contrib.rnn.BasicRNNCell(self.encoder_hidden_size)
            config_encoder_cell = tf.contrib.rnn.DropoutWrapper(config_encoder_cell, input_keep_prob=config['INPUT_DROPOUT'], output_keep_prob=config['OUTPUT_DROPOUT'])
            self.encoder_cell.append(config_encoder_cell)
        if not config['BIDIRECTIONAL_ENCODER']:
            self.encoder_cell = ExtendedMultiRNNCell(self.encoder_cell)

        print(self.encoder_cell)

        with tf.variable_scope("DynamicEncoder") as scope:
            if config['BIDIRECTIONAL_ENCODER']:
                self.encoder_fw_cell = copy.deepcopy(self.encoder_cell)
                self.encoder_bw_cell = copy.deepcopy(self.encoder_cell)
                # ((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_states, encoder_bw_states)) =                 tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_fw_cell, cell_bw=self.encoder_bw_cell, inputs=self.encoder_inputs_embedded, sequence_length=None, time_major=True, dtype=tf.float32)
                # print('\n\n\n\n\n\n\n', encoder_fw_outputs, '\n\n\n\n\n\n')
                # print('\n\n\n\n\n\n\n', encoder_fw_states, '\n\n\n\n\n\n')
                # self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
                # print('\n\n\n\n\n\n\n', self.encoder_outputs, '\n\n\n\n\n\n')


                # (self.encoder_outputs, encoder_fw_states, encoder_bw_states) =                 tf.nn.static_bidirectional_rnn(cell_fw=self.encoder_fw_cell, cell_bw=self.encoder_bw_cell, inputs=self.encoder_inputs_embedded, sequence_length=self.encoder_inputs_length, dtype=tf.float32)

                (self.encoder_outputs, encoder_fw_states, encoder_bw_states) =                 tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=self.encoder_fw_cell, cells_bw=self.encoder_bw_cell, inputs=tf.transpose(self.encoder_inputs_embedded,[1,0,2]), sequence_length=self.encoder_inputs_length, dtype=tf.float32)
                self.encoder_outputs = tf.transpose(self.encoder_outputs, [1,0,2])
                encoder_fw_state = encoder_fw_states[-1]
                encoder_bw_state = encoder_bw_states[-1]
                encoder_state_c = (encoder_fw_state.c+encoder_bw_state.c)/2.0
                encoder_state_h = (encoder_fw_state.h+encoder_bw_state.h)/2.0
                if config['USE_BS'] and not config['IS_TRAIN']:
                    self.encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
                    self.encoder_state = seq2seq.tile_batch(self.encoder_state, config['BEAM_WIDTH'])
                    self.encoder_outputs = tf.transpose(seq2seq.tile_batch(tf.transpose(self.encoder_outputs, [1,0,2]), config['BEAM_WIDTH']), [1,0,2])
                    self.encoder_inputs_length_att = seq2seq.tile_batch(self.encoder_inputs_length, config['BEAM_WIDTH'])
                else:
                    self.encoder_inputs_length_att = self.encoder_inputs_length
                self.encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
            else:
                self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(cell=self.encoder_cell, inputs=self.encoder_inputs_embedded, sequence_length=self.encoder_inputs_length, time_major=True, dtype=tf.float32)
                if config['USE_BS'] and not config['IS_TRAIN']:
                    self.encoder_outputs = tf.transpose(seq2seq.tile_batch(tf.transpose(self.encoder_outputs, [1,0,2]), config['BEAM_WIDTH']), [1,0,2])
                    self.encoder_inputs_length_att = seq2seq.tile_batch(self.encoder_inputs_length, config['BEAM_WIDTH'])
                else:
                    self.encoder_inputs_length_att = self.encoder_inputs_length
            print('Encoder Trainable Variables')
            self.encoder_variables = scope.trainable_variables()
            print(self.encoder_variables)






        self.decoder_cell = []
        for _ in range(config['DECODER_LAYERS']):
            config_decoder_cell = None
            if config['CELL'] in ['LSTM', 'lstm']:
                config_decoder_cell = tf.contrib.rnn.BasicLSTMCell(self.decoder_hidden_size)
            elif config['CELL'] in ['GRU', 'gru']:
                config_decoder_cell = tf.contrib.rnn.GRUCell(self.decoder_hidden_size)
            else:
                config_decoder_cell = tf.contrib.rnn.BasicRNNCell(self.decoder_hidden_size)
            config_decoder_cell = tf.contrib.rnn.DropoutWrapper(config_decoder_cell, input_keep_prob=config['INPUT_DROPOUT'], output_keep_prob=config['OUTPUT_DROPOUT'])
            self.decoder_cell.append(config_decoder_cell)

        # self.decoder_cell = ExtendedMultiRNNCell(self.decoder_cell)


        if config['ATTENTION_DECODER']:
            attention_cell = self.decoder_cell.pop(0)
            if config['ATTENTION_MECHANISE']=='LUONG':
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.decoder_hidden_size, tf.transpose(self.encoder_outputs,perm=[1,0,2]), memory_sequence_length=self.encoder_inputs_length_att, scale=True, name='shared_attention_mechanism')
                # self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(self.decoder_cell, attention_mechanism, attention_layer_size=self.decoder_hidden_size,  output_attention=True, name='shared_attention_decoder')
                attention_cell = tf.contrib.seq2seq.AttentionWrapper(attention_cell, attention_mechanism, attention_layer_size=self.decoder_hidden_size, output_attention=True)
            elif config['ATTENTION_MECHANISE']=='BAHDANAU':
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.decoder_hidden_size, tf.transpose(self.encoder_outputs,perm=[1,0,2]), memory_sequence_length=self.encoder_inputs_length_att, normalize=True)
                # self.decoder_cell = tf.contrib.seq2seq.AttentionWrapper(self.decoder_cell, attention_mechanism, attention_layer_size=self.decoder_hidden_size, output_attention=True, name='shared_attention_decoder')
                attention_cell = tf.contrib.seq2seq.AttentionWrapper(attention_cell, attention_mechanism, attention_layer_size=self.decoder_hidden_size, output_attention=True)
            else:
                raise Exception('config[\'ATTENTION_MECHANISE\'] should be LUONG or BAHDANAU')
            self.decoder_cell = GNMTAttentionMultiCell(attention_cell, self.decoder_cell)
        else:
            self.decoder_cell = ExtendedMultiRNNCell(self.decoder_cell)


        with tf.variable_scope("DynamicDecoder") as scope:

            initial_state = None

            if config['USE_BS'] and not config['IS_TRAIN']:
                initial_state = self.decoder_cell.zero_state(batch_size=self.batch_size*config['BEAM_WIDTH'], dtype=tf.float32)
                # if config['ATTENTION_DECODER']:
                #     cat_state = tuple([self.encoder_state] + list(initial_state.cell_state)[:-1])
                #     initial_state.clone(cell_state=cat_state)
                # else:
                #     initial_state = tuple([self.encoder_state] + list(initial_state[:-1]))
            else:
                initial_state = self.decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)

                # if config['ATTENTION_DECODER']:
                #     cat_state = tuple([self.encoder_state] + list(initial_state.cell_state)[:-1])
                #     initial_state.clone(cell_state=cat_state)
                # else:
                #     initial_state = tuple([self.encoder_state] + list(initial_state[:-1]))

            self.output_projection_layer = layers_core.Dense(self.output_size, use_bias=False)

            if config['IS_TRAIN']:
                self.train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_inputs_embedded, self.decoder_inputs_length, time_major=True)
                self.decoder_train=tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper=self.train_helper, initial_state=initial_state, output_layer=self.output_projection_layer)
                self.train_outputs, self.train_state, self.train_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(self.decoder_train, impute_finished=True, maximum_iterations=None)



            if config['USE_BS'] and not config['IS_TRAIN']:
                self.decoder_infer=tf.contrib.seq2seq.BeamSearchDecoder(self.decoder_cell, self.output_word_embedding_matrix, self.decoder_inputs[0], config['ID_END'], initial_state=initial_state, beam_width=config['BEAM_WIDTH'],                 output_layer=self.output_projection_layer)
            else:
                self.infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.output_word_embedding_matrix, self.decoder_inputs[0], config['ID_END'])
                self.decoder_infer=tf.contrib.seq2seq.BasicDecoder(self.decoder_cell, helper=self.infer_helper, initial_state=initial_state, output_layer=self.output_projection_layer)
            print("The decoder is:", self.decoder_infer)
            self.infer_outputs, self.infer_state, self.infer_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(self.decoder_infer, impute_finished=False, maximum_iterations=config['MAX_OUT_LEN']*2)
            if config['USE_BS'] and not config['IS_TRAIN']:
                self.infer_outputs = tf.transpose(self.infer_outputs.predicted_ids, [2,0,1])[0]
            else:
                self.infer_outputs = self.infer_outputs.sample_id


            if config['IS_TRAIN']:
                self.train_outputs = self.train_outputs.rnn_output

                self.train_loss = seq2seq.sequence_loss(logits=self.train_outputs, targets=tf.transpose(self.decoder_targets, perm=[1,0]), weights=self.decoder_targets_mask)


                self.rewards = tf.py_func(contentPenalty, [tf.transpose(self.encoder_inputs, perm=[1,0]), self.train_outputs, tf.constant(config['SRC_DICT'], dtype=tf.string), tf.constant(config['DST_DICT'], dtype=tf.string), tf.transpose(self.decoder_targets, perm=[1,0])], tf.float32)
                self.rewards.set_shape(self.train_outputs.get_shape())
                if config['RL_ENABLE']:
                    self.train_loss_rl = rlloss.sequence_loss_rl(logits=self.train_outputs, rewards=self.rewards, weights=self.decoder_targets_mask)
                else:
                    self.train_loss_rl = tf.constant(0.0)

                self.rewards_bleu = tf.py_func(bleuPenalty, [tf.transpose(self.encoder_inputs, perm=[1,0]), self.train_outputs, tf.constant(config['SRC_DICT'], dtype=tf.string), tf.constant(config['DST_DICT'], dtype=tf.string), tf.constant(config['HYP_FILE_PATH'], dtype=tf.string), tf.constant(config['REF_FILE_PATH_FORMAT'], dtype=tf.string)], tf.float32)
                self.rewards_bleu.set_shape(self.train_outputs.get_shape())
                if config['BLEU_RL_ENABLE']:
                    self.train_loss_rl_bleu = rlloss.sequence_loss_rl(logits=self.train_outputs, rewards=self.rewards_bleu, weights=self.decoder_targets_mask)/2
                else:
                    self.train_loss_rl_bleu = tf.constant(0.0)


                self.eval_loss = seq2seq.sequence_loss(logits=self.train_outputs, targets=tf.transpose(self.decoder_targets, perm=[1,0]), weights=self.decoder_targets_mask)

                if config['TRAIN_ON_EACH_STEP']:
                    self.final_loss = self.train_loss
                    if config['RL_ENABLE']:
                        self.final_loss = self.final_loss + self.train_loss_rl
                    if config['BLEU_RL_ENABLE']:
                        self.final_loss = self.final_loss + self.train_loss_rl_bleu
                else:
                    self.final_loss = self.eval_loss
            print('Decoder Trainable Variables')
            self.decoder_variables = scope.trainable_variables()
            print(self.decoder_variables)

        print('All Trainable Variables:')
        self.all_trainable_variables = tf.trainable_variables()
        print(self.all_trainable_variables)
        if config['IS_TRAIN']:
            def updateBP(loss, lr, var_list):

                # return [tf.contrib.layers.optimize_loss( loss=loss, global_step=self.global_step, learning_rate=lr[i], optimizer=self.optimizer, clip_gradients=config['CLIP_NORM'], variables=var_list[i], learning_rate_decay_fn=model_utils.create_learning_rate_decay_fn(decay_rate=1-config['LR_DECAY'], decay_steps=config['DECAY_STEPS'])) for i in range(len(lr))]
                gradients = [tf.gradients(loss, var_list[i]) for i in range(len(lr))]
                if config['CLIP']:
                    gradients = [tf.clip_by_global_norm(gradients[i], config['CLIP_NORM'])[0] for i in range(len(lr))]
                optimizers = [self.optimizer(lr[i]) for i in range(len(lr))]
                return [optimizers[i].apply_gradients(zip(gradients[i], var_list[i])) for i in range(len(lr))]

            if config['SPLIT_LR']:
                self.train_op = updateBP(self.final_loss, [self.word_embedding_learning_rate, self.encoder_learning_rate, self.decoder_learning_rate], [self.embedding_variables, self.encoder_variables, self.decoder_variables])
            else:
                self.train_op = updateBP(self.final_loss, [self.learning_rate], [self.all_trainable_variables])

        self.saver = model_utils.initGlobalSaver()

    def make_feed(self, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        return {
            self.encoder_inputs: encoder_inputs,
            self.encoder_inputs_length: encoder_inputs_length,
            self.encoder_inputs_mask: encoder_inputs_mask,
            self.decoder_inputs: decoder_inputs,
            self.decoder_inputs_length: decoder_inputs_length,
            self.decoder_inputs_mask: decoder_inputs_mask,
            self.decoder_targets: decoder_targets,
            self.decoder_targets_length: decoder_targets_length,
            self.decoder_targets_mask: decoder_targets_mask,
        }
    def train_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        train_feed = self.make_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        if self.rl_enable:
            if self.bleu_enable:
                [_, ce_loss, rl_loss, bleu_loss] = session.run([self.train_op, self.train_loss, self.train_loss_rl, self.train_loss_rl_bleu], train_feed)
                return [ce_loss, rl_loss, bleu_loss]
            else:
                [_, ce_loss, rl_loss] = session.run([self.train_op, self.train_loss, self.train_loss_rl], train_feed)
                return [ce_loss, rl_loss]
        else:
            [_, loss] = session.run([self.train_op, self.final_loss], train_feed)
            return loss

    def eval_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        infer_feed = self.make_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        [loss, outputs] = session.run([self.eval_loss, self.infer_outputs], infer_feed)
        return loss, outputs
    def test_on_batch(self, session, encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask):
        infer_feed = self.make_feed(encoder_inputs, encoder_inputs_length, encoder_inputs_mask, decoder_inputs, decoder_inputs_length, decoder_inputs_mask, decoder_targets, decoder_targets_length, decoder_targets_mask)
        [outputs] = session.run([self.infer_outputs], infer_feed)
        return outputs
def instanceOfInitModel(sess, config):
    ret = Seq2SeqModel(config)
    sess.run(tf.global_variables_initializer())
    print('Model Initialized.')
    return ret
