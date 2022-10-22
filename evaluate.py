import os
import re
import json
import copy
import numpy as np


def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def evaluation(result, args):
    truth_dir = os.path.join(args.data_dir, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir, exist_ok=True)

    fact_in_train_annotated = gen_train_facts(os.path.join(args.data_dir, "train_annotated.json"), truth_dir)
    # fact_in_train_distant = gen_train_facts(os.path.join(args.data_dir, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(args.data_dir, "dev.json")))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']

            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    tot_relations = len(std)

    tmp = copy.deepcopy(result)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    if tmp:
        submission_answer = [tmp[0]]
    else:
        submission_answer = []
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    # correct_in_train_distant = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    # if (n1['name'], n2['name'], r) in fact_in_train_distant:
                    #     in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            # if in_train_distant:
            #     correct_in_train_distant += 1

    re_p = 1.0 * correct_re / len(submission_answer) if len(submission_answer) > 0 else 0
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (
                len(submission_answer) - correct_in_train_annotated) if (
                len(submission_answer) - correct_in_train_annotated) > 0 else 0
    # re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (
    #             len(submission_answer) - correct_in_train_distant)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    # if re_p_ignore_train + re_r == 0:
    #     re_f1_ignore_train = 0
    # else:
    #     re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, re_f1_ignore_train_annotated


def eval_docred(args, total_pred, total_title, total_enum, id2rel, result_path, do_test=False):
    result = []
    f1, ign_f1 = 0., 0.

    rnum = len(id2rel)  # 96 without na
    for opred, title, enum in zip(total_pred, total_title, total_enum):
        pattern = re.compile(r'<extra_id_\d+>')
        pred = pattern.findall(opred)

        _pred = []
        for i in range(len(pred)):
            p = int(pred[i].replace('<extra_id_', '').replace('>', ''))

            if len(_pred) % 3 == 0:
                # head entity
                if 0 <= p - 101 < enum:
                    _pred.append(p - 101)
                elif p - 101 >= enum:
                    break
            elif len(_pred) % 3 == 1:
                # tail entity
                if 0 <= p - 101 < enum and p - 101 != _pred[-1]:
                    _pred.append(p - 101)
                elif p - 101 >= enum:
                    break
                else:
                    # 连续三元组，否则pop
                    _pred.pop()
            else:
                # relation
                if 0 <= p < rnum:  # [0, ..., 95]
                    _pred.append(id2rel[p])
                elif 0 <= p - 101 < enum:
                    # tail entity
                    _pred.pop(-2)
                    if p - 101 == _pred[-1]:
                        _pred.pop()
                    _pred.append(p - 101)
                else:
                    _pred.pop()
                    _pred.pop()

        tuple_pred = []
        for i in range(0, len(_pred), 3):
            if i == len(_pred) - 1 or i == len(_pred) - 2:
                break
            tuple_pred.append((_pred[i], _pred[i + 1], _pred[i + 2]))
        tuple_pred = list(set(tuple_pred))

        for p in tuple_pred:
            res = {'title': title,
                   'h_idx': p[0],
                   't_idx': p[1],
                   'r': p[2],
                   'evidence': []}
            result.append(res)

    with open(result_path, 'w') as f:
        json.dump(result, f)

    if not do_test:
        f1, ign_f1 = evaluation(result, args)
    return f1, ign_f1


def eval_f1(args, total_pred, total_ref, total_title, total_enum, total_filter_rel, na_id, result_path):
    dev_output = []
    tp, fp, fn = 0, 0, 0

    na = '<extra_id_' + str(na_id) + '>'
    for idx, (opred, ref, title, enum) in enumerate(zip(total_pred, total_ref, total_title, total_enum)):
        pattern = re.compile(r'<extra_id_\d+>')
        pred = pattern.findall(opred)

        if total_filter_rel:
            _tp, _fp, _fn = _eval(args, pred, ref.split(), na, enum, total_filter_rel[idx])
        elif args.cardinality == 4:
            _tp, _fp, _fn = _eval_4re(args, pred, ref.split(), na, enum)
        else:
            _tp, _fp, _fn = _eval(args, pred, ref.split(), na, enum)

        dev_output.append({'opred': opred,
                           'pred': ' '.join(pred),
                           'ref': ref,
                           'title': title,
                           'f': (_tp, _fp, _fn)})
        tp += _tp
        fp += _fp
        fn += _fn

    p = tp / (tp + fp) if tp + fp > 0 else 0
    r = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (p * r) / (p + r) if p + r > 0 else 0
    dev_output.append({'p': p, 'r': r, 'f': f1})
    with open(result_path, 'w') as f:
        json.dump(dev_output, f, indent=2)
    return p, r, f1


def _eval(args, pred, ref, na, enum, filter_rel=None):
    _pred = []
    for i in range(len(pred)):
        if pred[i] == na:
            if _pred:
                _pred.pop()
            if _pred:
                _pred.pop()
            continue
        _pred.append(pred[i])

    _ref = []
    for i in range(len(ref)):
        if ref[i] == na:
            _ref.pop()
            _ref.pop()
            continue
        _ref.append(ref[i])

    tp, fp, fn = 0, 0, 0
    used = [False] * len(_ref)
    for i in range(0, len(_pred), 3):
        if i == len(_pred) - 1 or i == len(_pred) - 2:
            break

        h_id = int(_pred[i].replace('<extra_id_', '').replace('>', ''))
        t_id = int(_pred[i + 1].replace('<extra_id_', '').replace('>', ''))
        if 'cdr' in args.data_dir.lower():
            assert len(tuple(enum)) == 2
            if h_id - 101 >= enum[0] or t_id - 126 >= enum[1]:
                continue
        elif 'gda' in args.data_dir.lower():
            assert len(tuple(enum)) == 2
            if h_id >= enum[0] or t_id - 50 >= enum[1]:
                continue
        elif 'scirex' in args.data_dir.lower():
            if args.end2end:
                assert isinstance(enum, int)
                if h_id >= enum or t_id >= enum:
                    continue
            else:
                assert len(tuple(enum)) == 4
                if h_id % 10 >= enum[h_id // 10] or (t_id < 30 and t_id % 10 >= enum[t_id // 10]) or \
                        (t_id >= 30 and t_id - 30 >= enum[3]):
                    continue
        else:
            assert isinstance(enum, int)
            if h_id - 101 >= enum or t_id - 101 >= enum:
                continue

        _find = False
        for j in range(0, len(_ref), 3):
            if _pred[i:i + 3] == _ref[j:j + 3] and not used[j]:
                tp += 1
                used[j] = True
                _find = True
                break
        if filter_rel:
            for _filter_rel in filter_rel:
                if _pred[i] == _filter_rel[0] and _pred[i + 1] == _filter_rel[1]:
                    _find = True
        if not _find:
            fp += 1
    for i in range(0, len(_ref), 3):
        if not used[i]:
            fn += 1
    return tp, fp, fn


def _eval_4re(args, pred, ref, na, enum):
    _pred = []
    for i in range(len(pred)):
        if pred[i] == na:
            if _pred:
                _pred.pop()
            if _pred:
                _pred.pop()
            if _pred:
                _pred.pop()
            if _pred:
                _pred.pop()
            continue
        _pred.append(pred[i])

    _ref = []
    for i in range(len(ref)):
        if ref[i] == na:
            _ref.pop()
            _ref.pop()
            _ref.pop()
            _ref.pop()
            continue
        _ref.append(ref[i])

    tp, fp, fn = 0, 0, 0
    used = [False] * len(_ref)
    for i in range(0, len(_pred), 5):
        if i == len(_pred) - 1 or i == len(_pred) - 2 or i == len(_pred) - 3 or i == len(_pred) - 4:
            break

        a_id = int(_pred[i].replace('<extra_id_', '').replace('>', ''))
        b_id = int(_pred[i + 1].replace('<extra_id_', '').replace('>', ''))
        c_id = int(_pred[i + 2].replace('<extra_id_', '').replace('>', ''))
        d_id = int(_pred[i + 3].replace('<extra_id_', '').replace('>', ''))
        if args.end2end:
            assert 'scirex' in args.data_dir.lower() and isinstance(enum, int)
            if a_id >= enum or b_id >= enum or c_id >= enum or d_id >= enum:
                continue
        else:
            assert 'scirex' in args.data_dir.lower() and len(tuple(enum)) == 4
            if a_id >= enum[0] or b_id - 10 >= enum[1] or c_id - 20 >= enum[2] or d_id - 30 >= enum[3]:
                continue

        _find = False
        for j in range(0, len(_ref), 5):
            if _pred[i:i + 5] == _ref[j:j + 5] and not used[j]:
                tp += 1
                used[j] = True
                _find = True
                break
        if not _find:
            fp += 1
    for i in range(0, len(_ref), 5):
        if not used[i]:
            fn += 1
    return tp, fp, fn


if __name__ == '__main__':
    # tp, fp, fn = 0, 0, 0
    #
    # na = '<extra_id_102>'
    # import argparse
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    # args.data_dir = '../data/GDA'
    # args.pretrained_model_name_or_path = 'GanjinZero/biobart-large'
    # args.max_input_length = 1024
    # args.max_label_length = 512
    # args.local_rank = -1
    #
    # from data import read_gda
    # from transformers import BartTokenizer
    # tokenizer = BartTokenizer.from_pretrained(args.pretrained_model_name_or_path,
    #                                           additional_special_tokens=[f"<extra_id_{i}>" for i in range(150)],
    #                                           add_prefix_space=True)
    # datasets = read_gda(args, ['test'], 'tmp_data_dir', tokenizer)
    #
    # jdata = json.load(open('outputs/biobart-gda/output_lr2e-05_warm0.02_maxl100_pen0.8/result/result_2022-06-11-06-52-02-480385_11.json', 'r'))
    # for truth, jline in zip(datasets['dev'], jdata):
    #     assert truth['title'] == jline['title']
    #     enum = truth['enum']
    #     opred = jline['opred']
    #     ref = jline['ref']
    #
    #     pattern = re.compile(r'<extra_id_\d+>')
    #     pred = pattern.findall(opred)
    #
    #     _tp, _fp, _fn = _eval(args, pred, ref.split(), na, enum)
    #
    #     tp += _tp
    #     fp += _fp
    #     fn += _fn
    #
    # p = tp / (tp + fp) if tp + fp > 0 else 0
    # r = tp / (tp + fn) if tp + fn > 0 else 0
    # f1 = 2 * (p * r) / (p + r) if p + r > 0 else 0
    # print(p, r, f1)
    jdata = json.load(open(os.path.join('dump/result_2022-06-23-07-00-26-025213_test_nbeams_1.json'), 'r'))
    with open('dump/result.json', 'w') as f:
        json.dump(jdata, f)
