import tensorflow as tf
from layers import initializer, regularizer, residual_block, highway, conv, mask_logits, \
                   trilinear, total_params, optimized_trilinear_for_attention, multihead_attention, \
                   biLSTM_layer_op
class Model(object):
    def __init__(self, config, batch, word_mat=None, char_mat=None, trainable=True, opt=True, demo = False, graph = None):
        self.config = config
        self.demo = demo
        self.graph = graph if graph is not None else tf.Graph()
        with self.graph.as_default():

            self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                               initializer=tf.constant_initializer(0), trainable=False)
            self.dropout = tf.placeholder_with_default(0.0, (), name="dropout")
            if self.demo:
                self.c = tf.placeholder(tf.int32, [None, config.test_para_limit],"context")
                self.q = tf.placeholder(tf.int32, [None, config.test_ques_limit],"question")
                self.ch = tf.placeholder(tf.int32, [None, config.test_para_limit, config.char_limit],"context_char")
                self.qh = tf.placeholder(tf.int32, [None, config.test_ques_limit, config.char_limit],"question_char")
                self.y1 = tf.placeholder(tf.int32, [None, config.test_para_limit],"answer_index1")
                self.y2 = tf.placeholder(tf.int32, [None, config.test_para_limit],"answer_index2")
            else:
                self.c, self.q, self.o1, self.o2, self.o3, self.o4, self.ch, self.qh, self.oh1, \
                self.oh2, self.oh3, self.oh4, self.qa_id, self.ans = batch.get_next()

            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
                word_mat, dtype=tf.float32), trainable=False)
            self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(
                char_mat, dtype=tf.float32))

            self.c_mask = tf.cast(self.c, tf.bool) # ?, 400
            self.q_mask = tf.cast(self.q, tf.bool)
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
            
            self.o1_mask = tf.cast(self.o1, tf.bool)
            self.o2_mask = tf.cast(self.o2, tf.bool)
            self.o3_mask = tf.cast(self.o3, tf.bool)
            self.o4_mask = tf.cast(self.o4, tf.bool)
            self.o1_len = tf.reduce_sum(tf.cast(self.o1_mask, tf.int32), axis=1)
            self.o2_len = tf.reduce_sum(tf.cast(self.o2_mask, tf.int32), axis=1)
            self.o3_len = tf.reduce_sum(tf.cast(self.o3_mask, tf.int32), axis=1)
            self.o4_len = tf.reduce_sum(tf.cast(self.o4_mask, tf.int32), axis=1)


            if opt:
                N, CL = config.batch_size if not self.demo else 1, config.char_limit
                self.c_maxlen = tf.reduce_max(self.c_len)
                self.q_maxlen = tf.reduce_max(self.q_len)

                self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen]) # 32,?
                self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen]) # 32,?
                self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
                self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])

                self.opt_maxlen = tf.reduce_max(tf.concat([self.o1_len, self.o2_len, self.o3_len, self.o4_len], 0))
                self.o1 = tf.slice(self.o1, [0, 0], [N, self.opt_maxlen])
                self.o2 = tf.slice(self.o2, [0, 0], [N, self.opt_maxlen])
                self.o3 = tf.slice(self.o3, [0, 0], [N, self.opt_maxlen])
                self.o4 = tf.slice(self.o4, [0, 0], [N, self.opt_maxlen])
                self.o1_mask = tf.slice(self.o1_mask, [0, 0], [N, self.opt_maxlen])
                self.o2_mask = tf.slice(self.o2_mask, [0, 0], [N, self.opt_maxlen])
                self.o3_mask = tf.slice(self.o3_mask, [0, 0], [N, self.opt_maxlen])
                self.o4_mask = tf.slice(self.o4_mask, [0, 0], [N, self.opt_maxlen])

                self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
                self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
                self.oh1 = tf.slice(self.oh1, [0, 0, 0], [N, self.opt_maxlen, CL])
                self.oh2 = tf.slice(self.oh2, [0, 0, 0], [N, self.opt_maxlen, CL])
                self.oh3 = tf.slice(self.oh3, [0, 0, 0], [N, self.opt_maxlen, CL])
                self.oh4 = tf.slice(self.oh4, [0, 0, 0], [N, self.opt_maxlen, CL])
            else:
                self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

            self.ch_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
            self.qh_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])
            self.oh1_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.oh1, tf.bool), tf.int32), axis=2), [-1])
            self.oh2_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.oh2, tf.bool), tf.int32), axis=2), [-1])
            self.oh3_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.oh3, tf.bool), tf.int32), axis=2), [-1])
            self.oh4_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.oh4, tf.bool), tf.int32), axis=2), [-1])


            self.forward()
            total_params()

            if trainable:
                self.lr = tf.minimum(config.learning_rate, 0.001 / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))
                self.opt = tf.train.AdamOptimizer(learning_rate = self.lr, beta1 = 0.8, beta2 = 0.999, epsilon = 1e-7)
                grads = self.opt.compute_gradients(self.loss)
                gradients, variables = zip(*grads)
                capped_grads, _ = tf.clip_by_global_norm(
                    gradients, config.grad_clip)
                self.train_op = self.opt.apply_gradients(
                    zip(capped_grads, variables), global_step=self.global_step)

    def forward(self):
        config = self.config
        N, PL, QL, OL, CL, d, dc, nh = config.batch_size if not self.demo else 1, self.c_maxlen, \
                                   self.q_maxlen, self.opt_maxlen, config.char_limit, \
                                   config.hidden, config.ta_c_dim, config.num_heads

        with tf.variable_scope("Input_Embedding_Layer"):
            def some_op(ch, L):
                _emb = tf.reshape(tf.nn.embedding_lookup(self.char_mat, ch), [N * L, CL, dc])
                _emb = tf.nn.dropout(_emb, 1.0 - 0.5 * self.dropout)
                _emb = conv(_emb, d, bias = True, activation = tf.nn.relu, \
                       kernel_size = 5, name = "char_conv", reuse = tf.AUTO_REUSE)
                _emb = tf.reduce_max(_emb, axis = 1)
                _emb = tf.reshape(_emb, [N, L, _emb.shape[-1]])
                return _emb

            ch_emb = some_op(self.ch, PL)
            qh_emb = some_op(self.qh, QL)
            oh1_emb = some_op(self.oh1, OL)
            oh2_emb = some_op(self.oh2, OL)
            oh3_emb = some_op(self.oh3, OL)
            oh4_emb = some_op(self.oh4, OL)

            def some_op2(word_emb, char_emb):
                # 32,?,300 batch,self.c_maxlen,emb_dim
                _emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, word_emb), 1.0 - self.dropout)
                _emb = tf.concat([_emb, char_emb], axis=2)
                _emb = highway(_emb, size = d, scope = "highway", dropout = self.dropout, reuse = tf.AUTO_REUSE)
                return _emb

            c_emb = some_op2(self.c, ch_emb)
            q_emb = some_op2(self.q, qh_emb)
            o1_emb = some_op2(self.o1, oh1_emb)
            o2_emb = some_op2(self.o2, oh2_emb)
            o3_emb = some_op2(self.o3, oh3_emb)
            o4_emb = some_op2(self.o4, oh4_emb)


        with tf.variable_scope("Embedding_Encoder_Layer", reuse = tf.AUTO_REUSE) as scope:
            # scope.reuse_variables()
            c = residual_block(c_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.c_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.c_len,
                scope = "Encoder_Residual_Block_o",
                bias = False,
                dropout = self.dropout)
            q = residual_block(q_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.q_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.q_len,
                scope = "Encoder_Residual_Block_o",
                reuse = tf.AUTO_REUSE, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)
            o1 = residual_block(o1_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.o1_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.o1_len,
                scope = "Encoder_Residual_Block_o",
                reuse = tf.AUTO_REUSE, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)
            o2 = residual_block(o2_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.o2_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.o2_len,
                scope = "Encoder_Residual_Block_o",
                reuse = tf.AUTO_REUSE, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)
            o3 = residual_block(o3_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.o3_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.o3_len,
                scope = "Encoder_Residual_Block_o",
                reuse = tf.AUTO_REUSE, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)
            o4 = residual_block(o4_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.o4_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.o4_len,
                scope = "Encoder_Residual_Block_o",
                reuse = tf.AUTO_REUSE, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)

        with tf.variable_scope("Context_to_Query_Attention_Layer", reuse=tf.AUTO_REUSE) as scope:
            S = optimized_trilinear_for_attention([c, q], self.c_maxlen, self.q_maxlen, input_keep_prob = 1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), axis= 1),(0,2,1))
            self.c2q = tf.matmul(S_, q)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]
        
            S = optimized_trilinear_for_attention([c, o1], self.c_maxlen, self.opt_maxlen, input_keep_prob = 1.0 - self.dropout)
            mask_o1 = tf.expand_dims(self.o1_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_o1))
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), axis = 1),(0,2,1))
            self.c2o1 = tf.matmul(S_, o1)
            self.o12c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_output0 = [c, self.c2o1, c * self.c2o1, c * self.o12c]

            S = optimized_trilinear_for_attention([c, o2], self.c_maxlen, self.opt_maxlen, input_keep_prob = 1.0 - self.dropout)
            mask_o2 = tf.expand_dims(self.o2_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_o2))
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), axis = 1),(0,2,1))
            self.c2o2 = tf.matmul(S_, o2)
            self.o22c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_output1 = [c, self.c2o2, c * self.c2o2, c * self.o22c]

            S = optimized_trilinear_for_attention([c, o3], self.c_maxlen, self.opt_maxlen, input_keep_prob = 1.0 - self.dropout)
            mask_o3 = tf.expand_dims(self.o3_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_o3))
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), axis = 1),(0,2,1))
            self.c2o3 = tf.matmul(S_, o3)
            self.o32c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_output2 = [c, self.c2o3, c * self.c2o3, c * self.o32c]

            S = optimized_trilinear_for_attention([c, o4], self.c_maxlen, self.opt_maxlen, input_keep_prob = 1.0 - self.dropout)
            mask_o4 = tf.expand_dims(self.o4_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_o4))
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), axis = 1),(0,2,1))
            self.c2o4 = tf.matmul(S_, o4)
            self.o42c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_output3 = [c, self.c2o4, c * self.c2o4, c * self.o42c]

        with tf.variable_scope("Model_Encoder_Layer", reuse = tf.AUTO_REUSE) as scope:
            # scope.reuse_variables()
            # attention_outputs : (N,?,96),(N,?,96),(N,?,96),(N,?,96)
            # inputs = (N,?,384)
            inputs = tf.concat(attention_outputs, axis = -1)
            self.enc = [conv(inputs, d, name = "input_projection")]
            for i in range(1):
                if i % 2 == 0: # dropout every 2 blocks
                    self.enc[i] = tf.nn.dropout(self.enc[i], 1.0 - self.dropout)
                self.enc.append(
                    residual_block(self.enc[i],
                        num_blocks = 7,
                        num_conv_layers = 2,
                        kernel_size = 5,
                        mask = self.c_mask,
                        num_filters = d,
                        num_heads = nh,
                        seq_len = self.c_len,
                        scope = "Model_Encoder",
                        bias = False,
                        reuse = True if i > 0 else None,
                        dropout = self.dropout)
                    )

        # with tf.variable_scope("Model_Encoder_Layer_Opt", reuse = tf.AUTO_REUSE) as scope:
            input0 = tf.concat(attention_output0, axis = -1)
            self.enc0 = [conv(input0, d, name = "input_projection")]
            self.enc0[0] = tf.nn.dropout(self.enc0[0], 1.0 - self.dropout)
            self.enc0.append(
                residual_block(self.enc0[0],
                    num_blocks = 7,
                    num_conv_layers = 2,
                    kernel_size = 5,
                    mask = self.c_mask,
                    num_filters = d,
                    num_heads = nh,
                    seq_len = self.c_len,
                    scope = "Model_Encoder",
                    bias = False,
                    reuse = tf.AUTO_REUSE,
                    dropout = self.dropout)
                )

            input1 = tf.concat(attention_output1, axis = -1)
            self.enc1 = [conv(input1, d, name = "input_projection")]
            self.enc1[0] = tf.nn.dropout(self.enc1[0], 1.0 - self.dropout)
            self.enc1.append(
                residual_block(self.enc1[0],
                    num_blocks = 7,
                    num_conv_layers = 2,
                    kernel_size = 5,
                    mask = self.c_mask,
                    num_filters = d,
                    num_heads = nh,
                    seq_len = self.c_len,
                    scope = "Model_Encoder",
                    bias = False,
                    reuse = True,
                    dropout = self.dropout)
                )

            input2 = tf.concat(attention_output2, axis = -1)
            self.enc2 = [conv(input2, d, name = "input_projection")]
            self.enc2[0] = tf.nn.dropout(self.enc2[0], 1.0 - self.dropout)
            self.enc2.append(
                residual_block(self.enc2[0],
                    num_blocks = 7,
                    num_conv_layers = 2,
                    kernel_size = 5,
                    mask = self.c_mask,
                    num_filters = d,
                    num_heads = nh,
                    seq_len = self.c_len,
                    scope = "Model_Encoder",
                    bias = False,
                    reuse = True,
                    dropout = self.dropout)
                )

            input3 = tf.concat(attention_output3, axis = -1)
            self.enc3 = [conv(input3, d, name = "input_projection")]
            self.enc3[0] = tf.nn.dropout(self.enc3[0], 1.0 - self.dropout)
            self.enc3.append(
                residual_block(self.enc3[0],
                    num_blocks = 7,
                    num_conv_layers = 2,
                    kernel_size = 5,
                    mask = self.c_mask,
                    num_filters = d,
                    num_heads = nh,
                    seq_len = self.c_len,
                    scope = "Model_Encoder",
                    bias = False,
                    reuse = True,
                    dropout = self.dropout)
                )

        with tf.variable_scope("Output_Layer") as scope:

            _q = biLSTM_layer_op(self.enc[1],N)
            _o1 = biLSTM_layer_op(self.enc0[1],N)
            _o2 = biLSTM_layer_op(self.enc1[1],N)
            _o3 = biLSTM_layer_op(self.enc2[1],N)
            _o4 = biLSTM_layer_op(self.enc3[1],N)

            '''
            _q  = tf.squeeze(conv(self.enc[1] , 1, bias = False, name = "linear_q"),-1)
            _o1 = tf.squeeze(conv(self.enc0[1], 1, bias = False, name = "linear_o"),-1)
            _o2 = tf.squeeze(conv(self.enc1[1], 1, bias = False, name = "linear_o", reuse = True),-1)
            _o3 = tf.squeeze(conv(self.enc2[1], 1, bias = False, name = "linear_o", reuse = True),-1)
            _o4 = tf.squeeze(conv(self.enc3[1], 1, bias = False, name = "linear_o", reuse = True),-1)

            _q  = tf.layers.dense(inputs=_q , units=1024, activation=tf.nn.leaky_relu, name='linear2_q', reuse=tf.AUTO_REUSE)
            _o1 = tf.layers.dense(inputs=_o1, units=1024, activation=tf.nn.leaky_relu, name='linear2_o', reuse=tf.AUTO_REUSE)
            _o2 = tf.layers.dense(inputs=_o2, units=1024, activation=tf.nn.leaky_relu, name='linear2_o', reuse=tf.AUTO_REUSE)
            _o3 = tf.layers.dense(inputs=_o3, units=1024, activation=tf.nn.leaky_relu, name='linear2_o', reuse=tf.AUTO_REUSE)
            _o4 = tf.layers.dense(inputs=_o4, units=1024, activation=tf.nn.leaky_relu, name='linear2_o', reuse=tf.AUTO_REUSE)

            _q  = tf.layers.dense(inputs=_q , units=64, activation=None, name='linear3_q', reuse=tf.AUTO_REUSE)
            _o1 = tf.layers.dense(inputs=_o1, units=64, activation=None, name='linear3_o', reuse=tf.AUTO_REUSE)
            _o2 = tf.layers.dense(inputs=_o2, units=64, activation=None, name='linear3_o', reuse=tf.AUTO_REUSE)
            _o3 = tf.layers.dense(inputs=_o3, units=64, activation=None, name='linear3_o', reuse=tf.AUTO_REUSE)
            _o4 = tf.layers.dense(inputs=_o4, units=64, activation=None, name='linear3_o', reuse=tf.AUTO_REUSE)
            '''

            o1_loss = tf.losses.cosine_distance(tf.nn.l2_normalize(_q, axis = -1), tf.nn.l2_normalize(_o1, axis = -1), axis = -1, reduction = tf.losses.Reduction.NONE)
            o2_loss = tf.losses.cosine_distance(tf.nn.l2_normalize(_q, axis = -1), tf.nn.l2_normalize(_o2, axis = -1), axis = -1, reduction = tf.losses.Reduction.NONE)
            o3_loss = tf.losses.cosine_distance(tf.nn.l2_normalize(_q, axis = -1), tf.nn.l2_normalize(_o3, axis = -1), axis = -1, reduction = tf.losses.Reduction.NONE)
            o4_loss = tf.losses.cosine_distance(tf.nn.l2_normalize(_q, axis = -1), tf.nn.l2_normalize(_o4, axis = -1), axis = -1, reduction = tf.losses.Reduction.NONE)

            _logits = tf.concat([o1_loss, o2_loss, o3_loss, o4_loss], axis = 1)

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=_logits, 
                labels=tf.one_hot(indices=tf.cast(self.ans, tf.int32), depth=4)))
            self.pred_ans = tf.argmin(_logits, axis = 1)

            ''' original
            start_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[2]],axis = -1),1, bias = False, name = "start_pointer"),-1)
            end_logits = tf.squeeze(conv(tf.concat([self.enc[1], self.enc[3]],axis = -1),1, bias = False, name = "end_pointer"), -1)
            self.logits = [mask_logits(start_logits, mask = self.c_mask),
                           mask_logits(end_logits, mask = self.c_mask)]

            logits1, logits2 = [l for l in self.logits]

            outer = tf.matmul(tf.expand_dims(tf.nn.softmax(logits1), axis=2),
                              tf.expand_dims(tf.nn.softmax(logits2), axis=1))
            outer = tf.matrix_band_part(outer, 0, config.ans_limit)
            self.yp1 = tf.argmax(tf.reduce_max(outer, axis=2), axis=1)
            self.yp2 = tf.argmax(tf.reduce_max(outer, axis=1), axis=1)
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits1, labels=self.y1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits2, labels=self.y2)
            self.loss = tf.reduce_mean(losses + losses2)
            '''

        if config.l2_norm is not None:
            variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            l2_loss = tf.contrib.layers.apply_regularization(regularizer, variables)
            self.loss += l2_loss

        if config.decay is not None:
            self.var_ema = tf.train.ExponentialMovingAverage(config.decay)
            ema_op = self.var_ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)

                self.assign_vars = []
                for var in tf.global_variables():
                    v = self.var_ema.average(var)
                    if v:
                        self.assign_vars.append(tf.assign(var,v))

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step
