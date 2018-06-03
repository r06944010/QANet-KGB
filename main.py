import tensorflow as tf
import ujson as json
import numpy as np
from tqdm import tqdm
import os
import csv

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''


from model import Model
from demo import Demo
from util import get_record_parser, convert_tokens, evaluate, get_batch_dataset, get_dataset


def train(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    # with open(config.char_emb_file, "r") as fh:
    #     char_mat = np.array(json.load(fh), dtype=np.float32)
    char_mat = None
    with open(config.train_eval_file, "r") as fh:
        train_eval_file = json.load(fh)
    with open(config.dev_eval_file, "r") as fh:
        dev_eval_file = json.load(fh)
    with open(config.dev_meta, "r") as fh:
        meta = json.load(fh)

    dev_total = meta["total"]

    print("Building model...")
    
    parser = get_record_parser(config) # parse is a function
    graph = tf.Graph()
    with graph.as_default() as g:
        train_dataset = get_batch_dataset(config.train_record_file, parser, config)
        dev_dataset = get_dataset(config.dev_record_file, parser, config)
        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types, train_dataset.output_shapes)
        train_iterator = train_dataset.make_one_shot_iterator() # iterates through it once
        dev_iterator = dev_dataset.make_one_shot_iterator()

        model = Model(config, iterator, word_mat, char_mat, graph = g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        patience = 0
        best_acc = 0.

        with tf.Session(config=sess_config) as sess:
            writer = tf.summary.FileWriter(config.log_dir)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            train_handle = sess.run(train_iterator.string_handle())
            dev_handle = sess.run(dev_iterator.string_handle())
            if os.path.exists(os.path.join(config.save_dir, "checkpoint")):
                saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            global_step = max(sess.run(model.global_step), 1)

            for _ in tqdm(range(global_step, config.num_steps + 1)):
                global_step = sess.run(model.global_step) + 1
                loss, train_op = sess.run([model.loss, model.train_op], feed_dict={
                                          handle: train_handle, model.dropout: config.dropout})
                if global_step % config.period == 0:
                    loss_sum = tf.Summary(value=[tf.Summary.Value(
                        tag="model/loss", simple_value=loss), ])
                    writer.add_summary(loss_sum, global_step)
                if global_step % config.checkpoint == 0:
                    print('---eval train---')
                    _, summ = evaluate_batch(
                        model, config.val_num_batches, train_eval_file, sess, "train", handle, train_handle)
                    for s in summ:
                        writer.add_summary(s, global_step)
                    print('---eval dev---')
                    metrics, summ = evaluate_batch(
                        model, dev_total // config.batch_size + 1, dev_eval_file, sess, "dev", handle, dev_handle)

                    dev_acc = metrics["acc"]
                    if dev_acc < best_acc:
                        patience += 1
                        if patience > config.early_stop:
                            break
                    else:
                        patience = 0
                        best_acc = max(best_acc, dev_acc)

                    for s in summ:
                        writer.add_summary(s, global_step)
                    writer.flush()
                    filename = os.path.join(
                        config.save_dir, "model_{}.ckpt".format(global_step))
                    saver.save(sess, filename)


def evaluate_batch(model, num_batches, eval_file, sess, data_type, handle, str_handle):
    print('evaluating batch ...')
    answer_dict = {}
    losses = []
    for _ in tqdm(range(1, num_batches + 1)):
        qa_id, loss, pred_ans = sess.run(
            [model.qa_id, model.loss, model.pred_ans], feed_dict={handle: str_handle})
        answer_dict_, _ = convert_tokens(
            eval_file, qa_id.tolist(), pred_ans)
        answer_dict.update(answer_dict_)
        losses.append(loss)
    loss = np.mean(losses)
    metrics = evaluate(eval_file, answer_dict)
    metrics["loss"] = loss
    loss_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/loss".format(data_type), simple_value=metrics["loss"]), ])
    acc_sum = tf.Summary(value=[tf.Summary.Value(
        tag="{}/acc".format(data_type), simple_value=metrics["acc"]), ])
    return metrics, [loss_sum, acc_sum]


def demo(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    char_mat = None
    #with open(config.char_emb_file, "r") as fh:
    #    char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    model = Model(config, None, word_mat, char_mat, trainable=False, demo = True)
    demo = Demo(model, config)


def test(config):
    with open(config.word_emb_file, "r") as fh:
        word_mat = np.array(json.load(fh), dtype=np.float32)
    char_mat = None
    #with open(config.char_emb_file, "r") as fh:
    #    char_mat = np.array(json.load(fh), dtype=np.float32)
    with open(config.test_eval_file, "r") as fh:
        eval_file = json.load(fh)
    with open(config.test_meta, "r") as fh:
        meta = json.load(fh)

    total = meta["total"]

    graph = tf.Graph()
    print("Loading model...")
    with graph.as_default() as g:
        test_batch = get_dataset(config.test_record_file, get_record_parser(
            config, is_test=True), config).make_one_shot_iterator()

        model = Model(config, test_batch, word_mat, char_mat, trainable=False, graph = g)

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(config.save_dir))
            if config.decay < 1.0:
                sess.run(model.assign_vars)
            losses = []
            answer_dict = {}
            remapped_dict = {}
            for step in tqdm(range(total // config.batch_size + 1)):
                qa_id, pred_ans = sess.run(
                    [model.qa_id, model.pred_ans])
                answer_dict_, _ = convert_tokens(
                    eval_file, qa_id.tolist(), pred_ans)
                answer_dict.update(answer_dict_)
            with open('pred_tocfl.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(['idx','ans'])
                for i in range(1,321):
                    writer.writerows([i,answer_dict[i]])

