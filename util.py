import tensorflow as tf
import re
from collections import Counter
import string

'''
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net
'''


def get_record_parser(config, is_test=False):
    def parse(example):
        para_limit = config.test_para_limit if is_test else config.para_limit
        ques_limit = config.test_ques_limit if is_test else config.ques_limit
        ans_limit = config.test_ans_limit if is_test else config.ans_limit
        char_limit = config.char_limit
        features = tf.parse_single_example(example,
                                           features={
                                               "context_idxs": tf.FixedLenFeature([], tf.string),
                                               "ques_idxs": tf.FixedLenFeature([], tf.string),
                                               "opt_idxs": tf.FixedLenFeature([], tf.string),
                                               "id": tf.FixedLenFeature([], tf.int64),
                                               "ans": tf.FixedLenFeature([], tf.int64)
                                           })
        context_idxs = tf.reshape(tf.decode_raw(
            features["context_idxs"], tf.int32), [para_limit])
        ques_idxs = tf.reshape(tf.decode_raw(
            features["ques_idxs"], tf.int32), [ques_limit])
        opt_idxs = tf.reshape(tf.decode_raw(
            features["opt_idxs"], tf.int32), [4, ans_limit])
        qa_id = features["id"]
        ans = features["ans"]
        return context_idxs, ques_idxs, opt_idxs[0], opt_idxs[1], opt_idxs[2], opt_idxs[3], qa_id, ans
    return parse


def get_batch_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).shuffle(config.capacity).repeat()
    if config.is_bucket:
        buckets = [tf.constant(num) for num in range(*config.bucket_range)]

        def key_func(context_idxs, ques_idxs, context_char_idxs, ques_char_idxs, y1, y2, qa_id):
            c_len = tf.reduce_sum(
                tf.cast(tf.cast(context_idxs, tf.bool), tf.int32))
            t = tf.clip_by_value(buckets, 0, c_len)
            return tf.argmax(t)

        def reduce_func(key, elements):
            return elements.batch(config.batch_size)

        dataset = dataset.apply(tf.contrib.data.group_by_window(
            key_func, reduce_func, window_size=5 * config.batch_size)).shuffle(len(buckets) * 25)
    else:
        dataset = dataset.batch(config.batch_size)
    return dataset


def get_dataset(record_file, parser, config):
    num_threads = tf.constant(config.num_threads, dtype=tf.int32)
    dataset = tf.data.TFRecordDataset(record_file).map(
        parser, num_parallel_calls=num_threads).repeat().batch(config.batch_size)
    return dataset


def convert_tokens(eval_file, qa_id, pred_ans):
    answer_dict = {}
    concon = {}
    for qid, ans in zip(qa_id, pred_ans):
        answer_dict[str(qid)] = ans
    return answer_dict, concon


def evaluate(eval_file, answer_dict):
    acc = total = 0
    for key, value in answer_dict.items():
        total += 1
        ground_truths = eval_file[key]["ans"]
        prediction = value
        if ground_truths == prediction:
            acc += 1
    acc = 100.0 * acc / total
    print('---accuarcy:---',acc)
    return {'acc': acc}


def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
