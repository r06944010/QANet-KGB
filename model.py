import tensorflow as tf
from layers import initializer, regularizer, residual_block, highway, conv, mask_logits, trilinear, total_params, optimized_trilinear_for_attention, multihead_attention

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
                self.c, self.q, self.r, self.w1, self.w2, self.w3, self.qa_id = batch.get_next()
                # print(self.c) # Tensor("IteratorGetNext:0", shape=(?, 400), dtype=int32)
                # print(self.q) # Tensor("IteratorGetNext:1", shape=(?, 50), dtype=int32)
                # print(self.r) # Tensor("IteratorGetNext:2", shape=(?, 50), dtype=int32)
                # self.c, self.q, self.ch, self.qh, self.y1, self.y2, self.qa_id = batch.get_next()

            # self.word_unk = tf.get_variable("word_unk", shape = [config.glove_dim], initializer=initializer())
            self.word_mat = tf.get_variable("word_mat", initializer=tf.constant(
                word_mat, dtype=tf.float32), trainable=False)
            '''
            self.char_mat = tf.get_variable("char_mat", initializer=tf.constant(
                char_mat, dtype=tf.float32)).
            '''

            self.c_mask = tf.cast(self.c, tf.bool) # ?, 400
            self.q_mask = tf.cast(self.q, tf.bool)
            self.c_len = tf.reduce_sum(tf.cast(self.c_mask, tf.int32), axis=1)
            self.q_len = tf.reduce_sum(tf.cast(self.q_mask, tf.int32), axis=1)
            
            self.r_mask = tf.cast(self.r, tf.bool)
            self.w1_mask = tf.cast(self.w1, tf.bool)
            self.w2_mask = tf.cast(self.w2, tf.bool)
            self.w3_mask = tf.cast(self.w3, tf.bool)
            self.r_len = tf.reduce_sum(tf.cast(self.r_mask, tf.int32), axis=1)
            self.w1_len = tf.reduce_sum(tf.cast(self.w1_mask, tf.int32), axis=1)
            self.w2_len = tf.reduce_sum(tf.cast(self.w2_mask, tf.int32), axis=1)
            self.w3_len = tf.reduce_sum(tf.cast(self.w3_mask, tf.int32), axis=1)


            if opt:
                N, CL = config.batch_size if not self.demo else 1, config.char_limit
                self.c_maxlen = tf.reduce_max(self.c_len)
                self.q_maxlen = tf.reduce_max(self.q_len)
                self.c = tf.slice(self.c, [0, 0], [N, self.c_maxlen]) # 32,?
                self.q = tf.slice(self.q, [0, 0], [N, self.q_maxlen]) # 32,?
                self.c_mask = tf.slice(self.c_mask, [0, 0], [N, self.c_maxlen])
                self.q_mask = tf.slice(self.q_mask, [0, 0], [N, self.q_maxlen])

                self.r_maxlen = tf.reduce_max(self.r_len)
                self.w1_maxlen = tf.reduce_max(self.w1_len)
                self.w2_maxlen = tf.reduce_max(self.w2_len)
                self.w3_maxlen = tf.reduce_max(self.w3_len)
                self.r = tf.slice(self.r, [0, 0], [N, self.r_maxlen])
                self.w1 = tf.slice(self.w1, [0, 0], [N, self.w1_maxlen])
                self.w2 = tf.slice(self.w2, [0, 0], [N, self.w2_maxlen])
                self.w3 = tf.slice(self.w3, [0, 0], [N, self.w3_maxlen])
                self.r_mask = tf.slice(self.r_mask, [0, 0], [N, self.r_maxlen])
                self.w1_mask = tf.slice(self.w1_mask, [0, 0], [N, self.w1_maxlen])
                self.w2_mask = tf.slice(self.w2_mask, [0, 0], [N, self.w2_maxlen])
                self.w3_mask = tf.slice(self.w3_mask, [0, 0], [N, self.w3_maxlen])

                '''
                self.ch = tf.slice(self.ch, [0, 0, 0], [N, self.c_maxlen, CL])
                self.qh = tf.slice(self.qh, [0, 0, 0], [N, self.q_maxlen, CL])
                self.y1 = tf.slice(self.y1, [0, 0], [N, self.c_maxlen])
                self.y2 = tf.slice(self.y2, [0, 0], [N, self.c_maxlen])
                '''
            else:
                self.c_maxlen, self.q_maxlen = config.para_limit, config.ques_limit

            '''
            self.ch_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.ch, tf.bool), tf.int32), axis=2), [-1])
            self.qh_len = tf.reshape(tf.reduce_sum(
                tf.cast(tf.cast(self.qh, tf.bool), tf.int32), axis=2), [-1])
            '''

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
        N, PL, QL, CL, d, dc, nh = config.batch_size if not self.demo else 1, self.c_maxlen, self.q_maxlen, config.char_limit, config.hidden, config.char_dim, config.num_heads

        with tf.variable_scope("Input_Embedding_Layer"):

            '''
            ch_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.ch), [N * PL, CL, dc])
            qh_emb = tf.reshape(tf.nn.embedding_lookup(
                self.char_mat, self.qh), [N * QL, CL, dc])
            ch_emb = tf.nn.dropout(ch_emb, 1.0 - 0.5 * self.dropout)
            qh_emb = tf.nn.dropout(qh_emb, 1.0 - 0.5 * self.dropout)

			# Bidaf style conv-highway encoder
            ch_emb = conv(ch_emb, d,
                bias = True, activation = tf.nn.relu, kernel_size = 5, name = "char_conv", reuse = None)
            qh_emb = conv(qh_emb, d,
                bias = True, activation = tf.nn.relu, kernel_size = 5, name = "char_conv", reuse = True)

            ch_emb = tf.reduce_max(ch_emb, axis = 1)
            qh_emb = tf.reduce_max(qh_emb, axis = 1)

            ch_emb = tf.reshape(ch_emb, [N, PL, ch_emb.shape[-1]])
            qh_emb = tf.reshape(qh_emb, [N, QL, ch_emb.shape[-1]])
            '''

            # 32,?,300 batch,self.c_maxlen,emb_dim
            c_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.c), 1.0 - self.dropout)
            # 32,?,300
            q_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.q), 1.0 - self.dropout)

            r_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.r), 1.0 - self.dropout)
            w1_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.w1), 1.0 - self.dropout)
            w2_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.w2), 1.0 - self.dropout)
            w3_emb = tf.nn.dropout(tf.nn.embedding_lookup(self.word_mat, self.w3), 1.0 - self.dropout)

            '''
            c_emb = tf.concat([c_emb, ch_emb], axis=2)
            q_emb = tf.concat([q_emb, qh_emb], axis=2)
            '''

            # 32,?,96
            c_emb = highway(c_emb, size = d, scope = "highway", dropout = self.dropout, reuse = None)
            # 32,?,96
            q_emb = highway(q_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)
            r_emb = highway(r_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)
            w1_emb = highway(w1_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)
            w2_emb = highway(w2_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)
            w3_emb = highway(w3_emb, size = d, scope = "highway", dropout = self.dropout, reuse = True)

        with tf.variable_scope("Embedding_Encoder_Layer"):
            c = residual_block(c_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.c_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.c_len,
                scope = "Encoder_Residual_Block",
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
                scope = "Encoder_Residual_Block",
                reuse = True, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)
            r = residual_block(r_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.r_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.r_len,
                scope = "Encoder_Residual_Block",
                reuse = True, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)
            w1 = residual_block(w1_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.w1_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.w1_len,
                scope = "Encoder_Residual_Block",
                reuse = True, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)
            w2 = residual_block(w2_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.w2_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.w2_len,
                scope = "Encoder_Residual_Block",
                reuse = True, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)
            w3 = residual_block(w3_emb,
                num_blocks = 1,
                num_conv_layers = 4,
                kernel_size = 7,
                mask = self.w3_mask,
                num_filters = d,
                num_heads = nh,
                seq_len = self.w3_len,
                scope = "Encoder_Residual_Block",
                reuse = True, # Share the weights between passage and question
                bias = False,
                dropout = self.dropout)

        with tf.variable_scope("Context_to_Query_Attention_Layer"):
            # C = tf.tile(tf.expand_dims(c,2),[1,1,self.q_maxlen,1])
            # Q = tf.tile(tf.expand_dims(q,1),[1,self.c_maxlen,1,1])
            # S = trilinear([C, Q, C*Q], input_keep_prob = 1.0 - self.dropout)
            S = optimized_trilinear_for_attention([c, q], self.c_maxlen, self.q_maxlen, input_keep_prob = 1.0 - self.dropout)
            mask_q = tf.expand_dims(self.q_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_q))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), axis= 1),(0,2,1))
            self.c2q = tf.matmul(S_, q)
            self.q2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_outputs = [c, self.c2q, c * self.c2q, c * self.q2c]

        with tf.variable_scope("Context_to_Right_Ans_Attention_Layer"):
            S = optimized_trilinear_for_attention([c, r], self.c_maxlen, self.r_maxlen, input_keep_prob = 1.0 - self.dropout)
            mask_r = tf.expand_dims(self.r_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_r))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), axis = 1),(0,2,1))
            self.c2r = tf.matmul(S_, r)
            self.r2c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_output0 = [c, self.c2r, c * self.c2r, c * self.r2c]

        with tf.variable_scope("Context_to_Wrong_1_Attention_Layer"):
            S = optimized_trilinear_for_attention([c, w1], self.c_maxlen, self.w1_maxlen, input_keep_prob = 1.0 - self.dropout)
            mask_w1 = tf.expand_dims(self.w1_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_w1))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), axis = 1),(0,2,1))
            self.c2w1 = tf.matmul(S_, w1)
            self.w12c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_output1 = [c, self.c2w1, c * self.c2w1, c * self.w12c]

        with tf.variable_scope("Context_to_Wrong_2_Attention_Layer"):
            S = optimized_trilinear_for_attention([c, w2], self.c_maxlen, self.w2_maxlen, input_keep_prob = 1.0 - self.dropout)
            mask_w2 = tf.expand_dims(self.w2_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_w2))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), axis = 1),(0,2,1))
            self.c2w2 = tf.matmul(S_, w2)
            self.w22c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_output2 = [c, self.c2w2, c * self.c2w2, c * self.w22c]

        with tf.variable_scope("Context_to_Wrong_3_Attention_Layer"):
            S = optimized_trilinear_for_attention([c, w3], self.c_maxlen, self.w3_maxlen, input_keep_prob = 1.0 - self.dropout)
            mask_w3 = tf.expand_dims(self.w3_mask, 1)
            S_ = tf.nn.softmax(mask_logits(S, mask = mask_w3))
            mask_c = tf.expand_dims(self.c_mask, 2)
            S_T = tf.transpose(tf.nn.softmax(mask_logits(S, mask = mask_c), axis = 1),(0,2,1))
            self.c2w3 = tf.matmul(S_, w3)
            self.w32c = tf.matmul(tf.matmul(S_, S_T), c)
            attention_output3 = [c, self.c2w3, c * self.c2w3, c * self.w32c]

        with tf.variable_scope("Model_Encoder_Layer", reuse = None):
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
        with tf.variable_scope("Model_Encoder_Layer", reuse = True):
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
                    dropout = self.dropout)
                )
        with tf.variable_scope("Model_Encoder_Layer", reuse = True):
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
                    dropout = self.dropout)
                )
        with tf.variable_scope("Model_Encoder_Layer", reuse = True):
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
                    dropout = self.dropout)
                )
        with tf.variable_scope("Model_Encoder_Layer", reuse = True):
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
                    dropout = self.dropout)
                )

        with tf.variable_scope("Output_Layer"):
            # do self attention
            '''
            _r = multihead_attention(self.enc[1], d, nh, memory=self.enc0[1], seq_len = self.c_len,
                bias = False, dropout = self.dropout)
            _w1 = multihead_attention(self.enc[1], d, nh, memory=self.enc1[1], seq_len = self.c_len,
                bias = False, dropout = self.dropout, reuse = True)
            _w2 = multihead_attention(self.enc[1], d, nh, memory=self.enc2[1], seq_len = self.c_len,
                bias = False, dropout = self.dropout, reuse = True)
            _w3 = multihead_attention(self.enc[1], d, nh, memory=self.enc3[1], seq_len = self.c_len,
                bias = False, dropout = self.dropout, reuse = True)
            '''
            _q = tf.squeeze(conv(self.enc[1], 1, bias = False, name = "linear"),-1)
            _r = tf.squeeze(conv(self.enc0[1], 1, bias = False, name = "linear", reuse = True),-1)
            _w1 = tf.squeeze(conv(self.enc1[1], 1, bias = False, name = "linear", reuse = True),-1)
            _w2 = tf.squeeze(conv(self.enc2[1], 1, bias = False, name = "linear", reuse = True),-1)
            _w3 = tf.squeeze(conv(self.enc3[1], 1, bias = False, name = "linear", reuse = True),-1)

            r_loss = tf.losses.cosine_distance(_q, _r, axis = -1, reduction = tf.losses.Reduction.NONE)
            w1_loss = tf.losses.cosine_distance(_q, _w1, axis = -1, reduction = tf.losses.Reduction.NONE)
            w2_loss = tf.losses.cosine_distance(_q, _w2, axis = -1, reduction = tf.losses.Reduction.NONE)
            w3_loss = tf.losses.cosine_distance(_q, _w3, axis = -1, reduction = tf.losses.Reduction.NONE)

            self.loss = tf.reduce_mean(r_loss - w1_loss - w2_loss - w3_loss)
            self.pred_ans = tf.argmin(tf.stack([r_loss, w1_loss, w2_loss, w3_loss]))

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
