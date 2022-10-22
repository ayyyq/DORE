import os
import json
import _pickle
import logging
from tqdm import tqdm
from itertools import combinations


def modify_sent(ostr, pos, e_id):
    op = 0
    st, ed = pos
    assert st != ed

    new_str = []
    cnt = 0
    for i in range(len(ostr)):
        if cnt == st and (ostr[i][0] != '<' or ostr[i] == '<'):  # '<e_id>' -> '<extra_id_{e_id}>'
            # 总插在后面
            new_str.append('<extra_id_' + str(e_id) + '>')
            op += 1
        if cnt == ed and op == 1:
            # 总插在前面
            new_str.append('<extra_id_' + str(e_id) + '>')
            op += 1

        new_str.append(ostr[i])
        if ostr[i][0] != '<' or ostr[i] == '<':
            cnt += 1
    if cnt == ed:
        new_str.append('<extra_id_' + str(e_id) + '>')
        op += 1
    if op != 2:
        print(' '.join(ostr))
    assert op == 2
    return new_str


def trans_entity_type(e_type):
    if e_type == 'PER':
        return 'person'
    if e_type == 'LOC':
        return 'location'
    if e_type == 'ORG':
        return 'organization'
    if e_type == 'TIME':
        return 'time'
    if e_type == 'NUM':
        return 'number'
    if e_type == 'MISC':
        return 'other'


def insert_extra_id(sents, entities):
    new_sents = []
    for i_s, sent in enumerate(sents):
        for i_t, token in enumerate(sent):
            new_token = [token]

            # start
            _add = []
            for i_e, ent in enumerate(entities):
                for i_m, m in enumerate(ent):
                    if i_s == m['sent_id'] and i_t == m['pos'][0]:
                        _add.append((m['pos'][1], i_e))
            if len(_add) > 1:
                _add = sorted(_add, key=lambda x: x[0], reverse=True)  # end越大越先添加
            for m in _add:
                new_token = ['<extra_id_' + str(m[1] + 101) + '>'] + new_token

            # end
            _add = []
            for i_e, ent in enumerate(entities):
                for i_m, m in enumerate(ent):
                    if i_s == m['sent_id'] and i_t == (m['pos'][1] - 1):
                        _add.append((m['pos'][0], i_e))
            if len(_add) > 1:
                _add = sorted(_add, key=lambda x: x[0], reverse=True)  # start越大越先添加
            for m in _add:
                new_token = new_token + ['<extra_id_' + str(m[1] + 101) + '>']
            new_sents.extend(new_token)
    return ' '.join(new_sents)


def read_docred(args, datasets, save_dir, tokenizer, refresh=False):
    # fout = open('t5.txt', 'a')
    # from transformers import T5Tokenizer
    # tok = T5Tokenizer.from_pretrained('t5-large',extra_ids=150)
    # max_length = 300
    rel2id = {}
    jrel = json.load(open(os.path.join(args.data_dir, 'rel_info.json'), 'r'))
    for key in jrel.keys():
        rel2id[key] = len(rel2id)
    rel2id['irrelevant'] = len(rel2id)

    ret = {}
    for dataset in datasets:
        cache_filepath = os.path.join(save_dir, dataset + '.pkl')
        if os.path.exists(cache_filepath) and not refresh:
            with open(cache_filepath, 'rb') as f:
                features = _pickle.load(f)
            if 'train' in dataset:
                ret['train'] = features
            else:
                ret[dataset] = features
            logging.info('Load ' + dataset + ' from ' + cache_filepath)
            continue

        jdata = json.load(open(os.path.join(args.data_dir, dataset + '.json'), 'r'))
        features = []
        for jline in jdata:
            sents = jline['sents']
            entities = jline['vertexSet']
            enum = len(entities)
            input_str = insert_extra_id(sents, entities)
            input_ids = tokenizer(input_str, truncation=True, max_length=args.max_input_length).input_ids

            full_labels = []
            for h in range(enum):
                full_labels.append([])
                for t in range(enum):
                    full_labels[-1].append([])
            if 'labels' in jline:
                for label in jline['labels']:
                    full_labels[label['h']][label['t']].append(rel2id[label['r']])

            lwin = 1
            label_str = []
            if 'labels' in jline:
                for h in range(enum):
                    for t in range(enum):
                        if h == t:
                            continue
                        if full_labels[h][t]:
                            for r in full_labels[h][t]:
                                label_str.append('<extra_id_' + str(h + 101) + '>')
                                label_str.append('<extra_id_' + str(t + 101) + '>')
                                label_str.append('<extra_id_' + str(r) + '>')
                        elif abs(h - t) <= lwin:
                            label_str.append('<extra_id_' + str(h + 101) + '>')
                            label_str.append('<extra_id_' + str(t + 101) + '>')
                            label_str.append('<extra_id_' + str(rel2id['irrelevant']) + '>')
            label_str = ' '.join(label_str)
            labels = tokenizer(label_str, truncation=True, max_length=args.max_label_length).input_ids

            # label_len = tok(label_str, return_tensors='pt').input_ids.shape[-1]
            # fout.write(label_str + '\t' + str(label_len) + '\n')

            features.append({'title': jline['title'],
                             'enum': enum,
                             'ref': label_str,
                             'input_ids': input_ids,
                             'labels': labels})

        if args.local_rank in [-1, 0]:
            with open(cache_filepath, 'wb') as f:
                _pickle.dump(features, f)
            if 'train' in dataset:
                ret['train'] = features
            else:
                ret[dataset] = features
            logging.info('Process ' + dataset + ' and save to ' + cache_filepath)
    # fout.close()
    return ret


def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res


def read_cdr(args, datasets, save_dir, tokenizer, refresh=False):
    # fout = open('cdr.txt', 'w')
    # max_row_enum = 16, max_col_enum = 17
    # max_input_length = 1024, max_label_length = 512, max_length = 150

    cdr_rel2id = {'1:CID:2': 0, '1:NR:2': 1}
    ret = {}
    for dataset in datasets:
        cache_filepath = os.path.join(save_dir, dataset + '.pkl')
        if os.path.exists(cache_filepath) and not refresh:
            with open(cache_filepath, 'rb') as f:
                features = _pickle.load(f)
            if 'train' in dataset:
                ret['train'] = features
            elif 'test' in dataset:
                ret['dev'] = features
            else:
                ret[dataset] = features
            logging.info('Load ' + dataset + ' from ' + cache_filepath)
            continue

        features = []
        pmids = set()
        with open(os.path.join(args.data_dir, dataset + '_filter.data'), 'r') as infile:
            lines = infile.readlines()
            for i_l, line in enumerate(tqdm(lines)):
                line = line.rstrip().split('\t')
                pmid = line[0]

                if pmid not in pmids:
                    pmids.add(pmid)
                    text = line[1]
                    prs = chunks(line[2:], 17)

                    ent2idx = {'Chemical': {}, 'Disease': {}}
                    entity_pos = {'Chemical': set(), 'Disease': set()}
                    for p in prs:
                        if p[0] == "not_include":
                            continue

                        eid = p[5]
                        es = list(map(int, p[8].split(':')))
                        ed = list(map(int, p[9].split(':')))
                        tpy = p[7]
                        for start, end in zip(es, ed):
                            entity_pos[tpy].add((eid, start, end))  # (eid, start, end)

                        eid = p[11]
                        es = list(map(int, p[14].split(':')))
                        ed = list(map(int, p[15].split(':')))
                        tpy = p[13]
                        for start, end in zip(es, ed):
                            entity_pos[tpy].add((eid, start, end))  # (eid, start, end)

                    for key in entity_pos.keys():
                        # 按照第一个mention出现的顺序给entity标号
                        entity_pos[key] = sorted(list(entity_pos[key]), key=lambda x: (x[1], x[2]))
                        for i_ep in range(len(entity_pos[key])):
                            eid = entity_pos[key][i_ep][0]
                            if eid not in ent2idx[key]:
                                ent2idx[key][eid] = len(ent2idx[key])
                            entity_pos[key][i_ep] = (ent2idx[key][eid], entity_pos[key][i_ep][1], entity_pos[key][i_ep][2])

                    sents = [t.split(' ') for t in text.split('|')]
                    # add <extra_id_10x> for each entity mentions
                    new_sents = []
                    i_t = 0
                    for sent in sents:
                        for token in sent:
                            tokens_wordpiece = [token]
                            for eid, start, end in entity_pos['Chemical']:
                                if i_t == start:
                                    tokens_wordpiece = ["<extra_id_" + str(eid + 101) + ">"] + tokens_wordpiece
                                if i_t + 1 == end:
                                    tokens_wordpiece = tokens_wordpiece + ["<extra_id_" + str(eid + 101) + ">"]
                            for eid, start, end in entity_pos['Disease']:
                                if i_t == start:
                                    tokens_wordpiece = ["<extra_id_" + str(eid + 126) + ">"] + tokens_wordpiece
                                if i_t + 1 == end:
                                    tokens_wordpiece = tokens_wordpiece + ["<extra_id_" + str(eid + 126) + ">"]
                            new_sents.extend(tokens_wordpiece)
                            i_t += 1
                    sents = ' '.join(new_sents)
                    input_ids = tokenizer(sents, truncation=True, max_length=args.max_input_length).input_ids

                    row_enum = len(ent2idx['Chemical'])
                    col_enum = len(ent2idx['Disease'])
                    full_labels = []
                    for h in range(row_enum):
                        full_labels.append([])
                        for t in range(col_enum):
                            full_labels[h].append([])
                    for p in prs:
                        if p[0] == "not_include":
                            continue
                        id1, typ1 = p[5], p[7]
                        id2, typ2 = p[11], p[13]
                        id1, id2 = ent2idx[typ1][id1], ent2idx[typ2][id2]

                        r = cdr_rel2id[p[0]]
                        assert typ1 != typ2
                        if typ1 == 'Chemical':
                            full_labels[id1][id2].append(r)
                        else:
                            full_labels[id2][id1].append(r)

                    label_str = []
                    filter_rel = []
                    for h in range(row_enum):
                        for t in range(col_enum):
                            if full_labels[h][t]:
                                assert len(full_labels[h][t]) == 1
                                for r in full_labels[h][t]:
                                    label_str.append('<extra_id_' + str(h + 101) + '>')
                                    label_str.append('<extra_id_' + str(t + 126) + '>')
                                    label_str.append('<extra_id_' + str(r) + '>')
                            else:
                                filter_rel.append(('<extra_id_' + str(h + 101) + '>', '<extra_id_' + str(t + 126) + '>'))
                    label_str = ' '.join(label_str)  # <head_id> <tail_id> relation
                    labels = tokenizer(label_str, truncation=True, max_length=args.max_label_length).input_ids

                    # fout.write(str(len(input_ids)) + '\t' + label_str + '\t' + str(len(labels)) + '\n')

                    features.append({'title': pmid,
                                     'enum': (row_enum, col_enum),
                                     'ref': label_str,
                                     'input_ids': input_ids,
                                     'labels': labels,
                                     'filter_rel': filter_rel})

        if args.local_rank in [-1, 0]:
            with open(cache_filepath, 'wb') as f:
                _pickle.dump(features, f)
            if 'train' in dataset:
                ret['train'] = features
            if 'test' in dataset:
                ret['dev'] = features
            else:
                ret[dataset] = features
            logging.info('Process ' + dataset + ' and save to ' + cache_filepath)

    # fout.close()
    return ret


def read_gda(args, datasets, save_dir, tokenizer, refresh=False):
    # fout = open('gda.txt', 'w')
    # max_row_enum = 33, max_col_enum = 18
    # max_input_length = 1024, max_label_length = 512, max_length = 100

    cdr_rel2id = {'1:GDA:2': 0, '1:NR:2': 1}
    ret = {}
    for dataset in datasets:
        cache_filepath = os.path.join(save_dir, dataset + '.pkl')
        if os.path.exists(cache_filepath) and not refresh:
            with open(cache_filepath, 'rb') as f:
                features = _pickle.load(f)
            if 'train' in dataset:
                ret['train'] = features
            elif 'test' in dataset:
                ret['dev'] = features
            else:
                ret[dataset] = features
            logging.info('Load ' + dataset + ' from ' + cache_filepath)
            continue

        features = []
        pmids = set()
        with open(os.path.join(args.data_dir, dataset + '.data'), 'r') as infile:
            lines = infile.readlines()
            for i_l, line in enumerate(tqdm(lines)):
                line = line.rstrip().split('\t')
                pmid = line[0]

                if pmid not in pmids:
                    pmids.add(pmid)
                    text = line[1]
                    prs = chunks(line[2:], 17)

                    ent2idx = {'Gene': {}, 'Disease': {}}
                    entity_pos = {'Gene': set(), 'Disease': set()}
                    for p in prs:
                        if p[0] == "not_include":
                            continue

                        eid = p[5]
                        es = list(map(int, p[8].split(':')))
                        ed = list(map(int, p[9].split(':')))
                        tpy = p[7]
                        for start, end in zip(es, ed):
                            entity_pos[tpy].add((eid, start, end))  # (eid, start, end)

                        eid = p[11]
                        es = list(map(int, p[14].split(':')))
                        ed = list(map(int, p[15].split(':')))
                        tpy = p[13]
                        for start, end in zip(es, ed):
                            entity_pos[tpy].add((eid, start, end))  # (eid, start, end)

                    for key in entity_pos.keys():
                        # 按照第一个mention出现的顺序给entity标号
                        entity_pos[key] = sorted(list(entity_pos[key]), key=lambda x: (x[1], x[2]))
                        for i_ep in range(len(entity_pos[key])):
                            eid = entity_pos[key][i_ep][0]
                            if eid not in ent2idx[key]:
                                ent2idx[key][eid] = len(ent2idx[key])
                            entity_pos[key][i_ep] = (ent2idx[key][eid], entity_pos[key][i_ep][1], entity_pos[key][i_ep][2])

                    sents = [t.split(' ') for t in text.split('|')]
                    # add <extra_id_10x> for each entity mentions
                    new_sents = []
                    i_t = 0
                    for sent in sents:
                        for token in sent:
                            tokens_wordpiece = [token]
                            for eid, start, end in entity_pos['Gene']:
                                if i_t == start:
                                    tokens_wordpiece = ["<extra_id_" + str(eid) + ">"] + tokens_wordpiece
                                if i_t + 1 == end:
                                    tokens_wordpiece = tokens_wordpiece + ["<extra_id_" + str(eid) + ">"]
                            for eid, start, end in entity_pos['Disease']:
                                if i_t == start:
                                    tokens_wordpiece = ["<extra_id_" + str(eid + 50) + ">"] + tokens_wordpiece
                                if i_t + 1 == end:
                                    tokens_wordpiece = tokens_wordpiece + ["<extra_id_" + str(eid + 50) + ">"]
                            new_sents.extend(tokens_wordpiece)
                            i_t += 1
                    sents = ' '.join(new_sents)
                    input_ids = tokenizer(sents, truncation=True, max_length=args.max_input_length).input_ids

                    row_enum = len(ent2idx['Gene'])
                    col_enum = len(ent2idx['Disease'])
                    full_labels = []
                    for h in range(row_enum):
                        full_labels.append([])
                        for t in range(col_enum):
                            full_labels[h].append([])
                    for p in prs:
                        if p[0] == "not_include":
                            continue
                        id1, typ1 = p[5], p[7]
                        id2, typ2 = p[11], p[13]
                        id1, id2 = ent2idx[typ1][id1], ent2idx[typ2][id2]

                        r = cdr_rel2id[p[0]]
                        assert typ1 != typ2
                        if typ1 == 'Gene':
                            full_labels[id1][id2].append(r)
                        else:
                            full_labels[id2][id1].append(r)

                    label_str = []
                    for h in range(row_enum):
                        for t in range(col_enum):
                            assert len(full_labels[h][t]) == 1
                            if full_labels[h][t]:
                                for r in full_labels[h][t]:
                                    label_str.append('<extra_id_' + str(h) + '>')
                                    label_str.append('<extra_id_' + str(t + 50) + '>')
                                    label_str.append('<extra_id_' + str(r + 101) + '>')
                    label_str = ' '.join(label_str)  # <head_id> <tail_id> relation
                    labels = tokenizer(label_str, truncation=True, max_length=args.max_label_length).input_ids

                    # fout.write(str(len(input_ids)) + '\t' + label_str + '\t' + str(len(labels)) + '\n')

                    features.append({'title': pmid,
                                     'enum': (row_enum, col_enum),
                                     'ref': label_str,
                                     'input_ids': input_ids,
                                     'labels': labels})

        if args.local_rank in [-1, 0]:
            with open(cache_filepath, 'wb') as f:
                _pickle.dump(features, f)
            if 'train' in dataset:
                ret['train'] = features
            if 'test' in dataset:
                ret['dev'] = features
            else:
                ret[dataset] = features
            logging.info('Process ' + dataset + ' and save to ' + cache_filepath)

    # fout.close()
    return ret


def generate_entities_using_gold_inputs(doc, used_entities):
    # when using gold inputs, we know the coreferences and entity types and merge method subrelations
    true_entities = {}
    for typ in used_entities:
        true_entities[typ] = set([r[typ] for r in doc['n_ary_relations']])
    method_subrelations = doc['method_subrelations']
    for m, subnames in method_subrelations.items():  # TODO: whether to retain
        for sm in subnames :
            if m != sm[1] :
                doc['coref'][m] += doc['coref'][sm[1]]

    entities = {}
    ent2idx = {}
    for typ in true_entities.keys():
        entities[typ] = {}
        ent2idx[typ] = {}
        for e in list(true_entities[typ]):
            if len(doc["coref"][e]) > 0:
                entities[typ][e] = list(set([tuple(x) for x in doc['coref'][e]]))
                entities[typ][e] = sorted(entities[typ][e], key=lambda x: (x[0], x[1]))
        entities[typ] = dict(sorted(entities[typ].items(), key=lambda x: (x[1][0][0], x[1][0][1])))  # 按照第一个mention出现的顺序给entity标号
        for e in entities[typ].keys():
            assert e not in ent2idx[typ]
            ent2idx[typ][e] = len(ent2idx[typ])
    return entities, ent2idx


def generate_relations_using_gold_inputs(doc, ent2idx, cardinality, used_entities):
    typ2id = {typ: idx for idx, typ in enumerate(used_entities)}
    _relations = []
    for types in combinations(used_entities, cardinality):
        relations = [tuple((t, x[t]) for t in types) for x in doc['n_ary_relations']]

        # make sure each entity has at least one cluster and make (entity_1, entity_2, relation) unique
        relations = set([x for x in relations if has_all_mentions(doc, x)])

        for relation in relations:
            _relation = [0] * cardinality
            for i_e, entity in enumerate(relation):
                entity_type, entity_name = entity
                assert entity_name in ent2idx[entity_type]
                _relation[i_e] = ent2idx[entity_type][entity_name] + typ2id[entity_type] * 10
            _relations.append(tuple(_relation))
    return _relations


def has_all_mentions(doc, relation):
    # Make sure each entity has at least one mention.
    has_mentions = all(len(doc["coref"][x[1]]) > 0 for x in relation)
    return has_mentions


def read_scirex_using_gold_inputs(args, datasets, save_dir, tokenizer, refresh=False):
    # fout = open('scirex.txt', 'w')
    # max_input_length = 4096, max_label_length = 200, max_length = 150

    # map_available_entity_to_true = {"Material": "dataset", "Method": "model_name", "Metric": "metric", "Task": "task"}
    map_available_entity_to_true = {"Task": "task", "Metric": "metric", "Method": "model_name", "Material": "dataset"}
    # [7, 6, 9, 49]
    used_entities = list(map_available_entity_to_true.keys())
    cardinality = args.cardinality
    rel2id = {'relevant': 0, 'irrelevant': 1}

    ret = {}
    for dataset in datasets:
        cache_filepath = os.path.join(save_dir, dataset + '.pkl')
        if os.path.exists(cache_filepath) and not refresh:
            with open(cache_filepath, 'rb') as f:
                features = _pickle.load(f)
            if 'train' in dataset:
                ret['train'] = features
            elif 'test' in dataset:
                ret['dev'] = features
            else:
                ret[dataset] = features
            logging.info('Load ' + dataset + ' from ' + cache_filepath)
            continue

        with open(os.path.join(args.data_dir, dataset + '.jsonl'), 'r') as f:
            input_data = [json.loads(l) for l in f.readlines()]
        features = []
        for input_doc in tqdm(input_data):
            entities, ent2idx = generate_entities_using_gold_inputs(input_doc, used_entities)

            tokens = input_doc['words']
            new_tokens = []
            i_t = 0
            for token in tokens:
                tokens_wordpiece = [token]
                for i_typ, typ in enumerate(entities.keys()):
                    for e, entity_pos in entities[typ].items():
                        for start, end in entity_pos:
                            if i_t == start:
                                tokens_wordpiece = ["<extra_id_" + str(ent2idx[typ][e] + i_typ * 10) + ">"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["<extra_id_" + str(ent2idx[typ][e] + i_typ * 10) + ">"]
                new_tokens.extend(tokens_wordpiece)
                i_t += 1
            input_str = ' '.join(new_tokens)
            input_ids = tokenizer(input_str, truncation=True, max_length=args.max_input_length).input_ids

            # (Task, Method, Material, Metric, 'relevant'/'irrelevant')
            relations = generate_relations_using_gold_inputs(input_doc, ent2idx, cardinality, used_entities)

            # lwin = 1
            enum = []
            eid_list = []
            for i_typ, _eid_list in enumerate(ent2idx.values()):
                enum.append(len(_eid_list))
                if cardinality == 2:
                    eid_list.extend([eid + i_typ * 10 for eid in _eid_list.values()])
                else:
                    eid_list.append([eid + i_typ * 10 for eid in _eid_list.values()])

            label_str = []
            if cardinality == 2:
                for i_h, h in enumerate(eid_list):
                    if h // 10 >= 3:
                        break
                    for i_t, t in enumerate(eid_list[i_h + 1:]):
                        if t // 10 == h // 10:
                            continue
                        if (h, t) in relations:
                            label_str.append('<extra_id_' + str(h) + '>')
                            label_str.append('<extra_id_' + str(t) + '>')
                            label_str.append('<extra_id_' + str(rel2id['relevant'] + 101) + '>')
                        else:
                            label_str.append('<extra_id_' + str(h) + '>')
                            label_str.append('<extra_id_' + str(t) + '>')
                            label_str.append('<extra_id_' + str(rel2id['irrelevant'] + 101) + '>')
            else:
                assert len(eid_list) == 4
                for a in eid_list[0]:
                    for b in eid_list[1]:
                        for c in eid_list[2]:
                            for d in eid_list[3]:
                                label_str.append('<extra_id_' + str(a) + '>')
                                label_str.append('<extra_id_' + str(b) + '>')
                                label_str.append('<extra_id_' + str(c) + '>')
                                label_str.append('<extra_id_' + str(d) + '>')
                                if (a, b, c, d) in relations:
                                    label_str.append('<extra_id_' + str(rel2id['relevant'] + 101) + '>')
                                else:
                                    label_str.append('<extra_id_' + str(rel2id['irrelevant'] + 101) + '>')
            label_str = ' '.join(label_str)
            labels = tokenizer(label_str, truncation=True, max_length=args.max_label_length).input_ids

            # fout.write(str(len(input_ids)) + '\t' + label_str + '\t' + str(len(labels)) + '\n')

            features.append({'title': input_doc['doc_id'],
                             'enum': enum,
                             'ref': label_str,
                             'input_ids': input_ids,
                             'labels': labels})

        if args.local_rank in [-1, 0]:
            with open(cache_filepath, 'wb') as f:
                _pickle.dump(features, f)
            if 'train' in dataset:
                ret['train'] = features
            if 'test' in dataset:
                ret['dev'] = features
            else:
                ret[dataset] = features
            logging.info('Process ' + dataset + ' and save to ' + cache_filepath)
    # fout.close()
    return ret


def generate_entities(doc, used_entities):
    entities = {entity_name: sorted(_coref, key=lambda x: (x[0], x[1])) for entity_name, _coref in doc['coref'].items()
                if len(_coref) > 0}
    entities = dict(sorted(entities.items(), key=lambda x: (x[1][0][0], x[1][0][1])))  # 按照第一个mention出现的顺序给entity标号
    ent2idx = {e: idx for idx, e in enumerate(entities.keys())}
    return entities, ent2idx


def generate_true_entities(doc, used_entities):
    true_entities = set([r[e] for r in doc['n_ary_relations'] for e in used_entities])
    entities = {}
    for e in list(true_entities):
        if len(doc['coref'][e]) > 0:
            entities[e] = sorted(doc['coref'][e], key=lambda x: (x[0], x[1]))
    entities = dict(sorted(entities.items(), key=lambda x: (x[1][0][0], x[1][0][1])))  # 按照第一个mention出现的顺序给entity标号
    ent2idx = {e: idx for idx, e in enumerate(entities.keys())}
    return entities, ent2idx


def generate_relations(doc, rel2id, ent2idx, cardinality, used_entities):
    enum = len(ent2idx)
    full_labels = []
    if cardinality == 2:
        for h in range(enum):
            full_labels.append([])
            for t in range(enum):
                full_labels[-1].append([])

    for types in combinations(used_entities, cardinality):
        rel_typ = ' and '.join(types).lower()

        relations = [tuple((t, x[t]) for t in types) for x in doc['n_ary_relations']]

        # make sure each entity has at least one cluster and make (entity_1, entity_2, relation) unique
        relations = set([x for x in relations if has_all_mentions(doc, x)])

        for relation in relations:
            if cardinality == 2:
                e0 = ent2idx[relation[0][1]]
                e1 = ent2idx[relation[1][1]]
                full_labels[e0][e1].append(rel2id[rel_typ])
                full_labels[e1][e0].append(rel2id[rel_typ])
            else:
                e_set = set([ent2idx[rel[1]] for rel in relation])
                full_labels.append(e_set)
    return full_labels


def read_scirex(args, datasets, save_dir, tokenizer, refresh=False):
    # fout = open('scirex-4re-full-all.txt', 'w')
    # max_input_length = 4096, max_label_length = 300/500, max_length = 250/150

    map_available_entity_to_true = {"Material": "dataset", "Metric": "metric", "Task": "task", "Method": "model_name"}
    used_entities = list(map_available_entity_to_true.keys())
    cardinality = args.cardinality
    rel2id = {}
    for types in combinations(used_entities, cardinality):
        rel_typ = ' and '.join(types).lower()
        rel2id[rel_typ] = len(rel2id)
    rel2id['irrelevant'] = len(rel2id)

    ret = {}
    for dataset in datasets:
        cache_filepath = os.path.join(save_dir, dataset + '.pkl')
        if os.path.exists(cache_filepath) and not refresh:
            with open(cache_filepath, 'rb') as f:
                features = _pickle.load(f)
            if 'train' in dataset:
                ret['train'] = features
            elif 'test' in dataset:
                ret['dev'] = features
            else:
                ret[dataset] = features
            logging.info('Load ' + dataset + ' from ' + cache_filepath)
            continue

        with open(os.path.join(args.data_dir, dataset + '.jsonl'), 'r') as f:
            input_data = [json.loads(l) for l in f.readlines()]
        features = []
        for input_doc in tqdm(input_data):
            entities, ent2idx = generate_entities(input_doc, used_entities)

            tokens = input_doc['words']
            new_tokens = []
            i_t = 0
            for token in tokens:
                tokens_wordpiece = [token]
                for e, entity_pos in entities.items():
                    for start, end in entity_pos:
                        if i_t == start:
                            tokens_wordpiece = ["<extra_id_" + str(ent2idx[e]) + ">"] + tokens_wordpiece
                        if i_t + 1 == end:
                            tokens_wordpiece = tokens_wordpiece + ["<extra_id_" + str(ent2idx[e]) + ">"]
                new_tokens.extend(tokens_wordpiece)
                i_t += 1
            input_str = ' '.join(new_tokens)
            input_ids = tokenizer(input_str, truncation=True, max_length=args.max_input_length).input_ids

            full_labels = generate_relations(input_doc, rel2id, ent2idx, cardinality, used_entities)

            enum = len(ent2idx)
            lwin = 1
            label_str = []
            if cardinality == 2:
                for h in range(enum):
                    for t in range(h + 1, enum):
                        if h == t:
                            continue
                        if full_labels[h][t]:
                            for r in full_labels[h][t]:
                                label_str.append('<extra_id_' + str(h) + '>')
                                label_str.append('<extra_id_' + str(t) + '>')
                                label_str.append('<extra_id_' + str(r + 101) + '>')
                        else:
                            label_str.append('<extra_id_' + str(h) + '>')
                            label_str.append('<extra_id_' + str(t) + '>')
                            label_str.append('<extra_id_' + str(rel2id['irrelevant'] + 101) + '>')
            else:
                for a in range(enum):
                    for b in range(a + 1, enum):
                        for c in range(b + 1, enum):
                            for d in range(c + 1, enum):
                                if {a, b, c, d} in full_labels:
                                    label_str.append('<extra_id_' + str(a) + '>')
                                    label_str.append('<extra_id_' + str(b) + '>')
                                    label_str.append('<extra_id_' + str(c) + '>')
                                    label_str.append('<extra_id_' + str(d) + '>')
                                    label_str.append('<extra_id_' + str(101) + '>')
                                elif abs(a - b) <= lwin and abs(b - c) <= lwin and abs(c - d) <= lwin:
                                    label_str.append('<extra_id_' + str(a) + '>')
                                    label_str.append('<extra_id_' + str(b) + '>')
                                    label_str.append('<extra_id_' + str(c) + '>')
                                    label_str.append('<extra_id_' + str(d) + '>')
                                    label_str.append('<extra_id_' + str(102) + '>')
            label_str = ' '.join(label_str)
            labels = tokenizer(label_str, truncation=True, max_length=args.max_label_length).input_ids

            # fout.write(label_str + '\t' + str(len(labels)) + '\n')

            features.append({'title': input_doc['doc_id'],
                             'enum': enum,
                             'ref': label_str,
                             'input_ids': input_ids,
                             'labels': labels})

        if args.local_rank in [-1, 0]:
            with open(cache_filepath, 'wb') as f:
                _pickle.dump(features, f)
            if 'train' in dataset:
                ret['train'] = features
            if 'test' in dataset:
                ret['dev'] = features
            else:
                ret[dataset] = features
            logging.info('Process ' + dataset + ' and save to ' + cache_filepath)
    # fout.close()
    return ret


def read_scirex_test(args, pred_path, save_dir, tokenizer, refresh=False):
    # fout = open('scirex-true-all.txt', 'w')
    # max_input_length = 4096, max_label_length = 300, max_length = 250

    map_available_entity_to_true = {"Material": "dataset", "Metric": "metric", "Task": "task", "Method": "model_name"}
    used_entities = list(map_available_entity_to_true.keys())
    cardinality = args.cardinality
    rel2id = {}
    for types in combinations(used_entities, cardinality):
        rel_typ = ' and '.join(types).lower()
        rel2id[rel_typ] = len(rel2id)
    rel2id['irrelevant'] = len(rel2id)

    ret = {}
    with open(pred_path, 'r') as f:
        pred_data = [json.loads(l) for l in f.readlines()]
    with open(os.path.join(args.data_dir, 'test.jsonl'), 'r') as f:
        input_data = [json.loads(l) for l in f.readlines()]
    assert len(pred_data) == len(input_data)

    features = []
    for pred_doc, input_doc in tqdm(zip(pred_data, input_data)):
        assert pred_doc['doc_key'].split('_')[0] == input_doc['doc_id']

        predicted_clusters = []
        for _predicted_clusters in pred_doc['predicted_clusters']:
            orig_predicted_clusters = []
            for start, end in _predicted_clusters:
                orig_predicted_clusters.append([pred_doc['subtoken_map'][start], pred_doc['subtoken_map'][end] + 1])
            orig_predicted_clusters = sorted(orig_predicted_clusters, key=lambda x: (x[0], x[1]))
            predicted_clusters.append(orig_predicted_clusters)
        predicted_clusters = sorted(predicted_clusters, key=lambda x: (x[0][0], x[0][1]))

        # input_str
        tokens = input_doc['words']
        new_tokens = []
        i_t = 0
        for token in tokens:
            tokens_wordpiece = [token]
            for i_e, entity_pos in enumerate(predicted_clusters):
                for start, end in entity_pos:
                    if i_t == start:
                        tokens_wordpiece = ["<extra_id_" + str(i_e) + ">"] + tokens_wordpiece
                    if i_t + 1 == end:
                        tokens_wordpiece = tokens_wordpiece + ["<extra_id_" + str(i_e) + ">"]
            new_tokens.extend(tokens_wordpiece)
            i_t += 1
        input_str = ' '.join(new_tokens)
        input_ids = tokenizer(input_str, truncation=True, max_length=args.max_input_length).input_ids

        entities, ent2idx = generate_entities(input_doc, used_entities)
        ent2cluster = {}
        for entity_name, _coref in entities.items():
            ent2cluster[entity_name] = (-1, 0)
            _coref = set(tuple(x) for x in _coref)
            for i_pred, pred_cluster in enumerate(predicted_clusters):
                pred_cluster = set(tuple(x) for x in pred_cluster)
                intersec = len(set(_coref) & set(pred_cluster))
                if intersec > ent2cluster[entity_name][1]:
                    ent2cluster[entity_name] = (i_pred, intersec)

        # relation
        enum = len(predicted_clusters)
        full_labels = []
        extra_labels = []
        if cardinality == 2:
            for h in range(enum):
                full_labels.append([])
                for t in range(enum):
                    full_labels[-1].append([])

        for types in combinations(used_entities, cardinality):
            rel_typ = ' and '.join(types).lower()

            relations = [tuple((t, x[t]) for t in types) for x in input_doc['n_ary_relations']]

            # make sure each entity has at least one cluster and make (entity_1, entity_2, relation) unique
            relations = set([x for x in relations if has_all_mentions(input_doc, x)])

            for relation in relations:
                if cardinality == 2:
                    e0 = ent2cluster[relation[0][1]][0]
                    e1 = ent2cluster[relation[1][1]][0]
                    if e0 != -1 and e1 != -1:
                        full_labels[e0][e1].append(rel2id[rel_typ])
                        full_labels[e1][e0].append(rel2id[rel_typ])
                    else:
                        extra_labels += ['<extra_id_' + str(rel2id[rel_typ] + 101) + '>'] * 3
                else:
                    e_set = set([ent2cluster[rel[1]][0] for rel in relation])
                    if -1 not in e_set and len(e_set) == 4:
                        full_labels.append(e_set)
                    else:
                        extra_labels += ['<extra_id_' + str(rel2id[rel_typ] + 101) + '>'] * 5


        label_str = []
        if cardinality == 2:
            for h in range(enum):
                for t in range(h + 1, enum):
                    if h == t:
                        continue
                    if full_labels[h][t]:
                        for r in full_labels[h][t]:
                            label_str.append('<extra_id_' + str(h) + '>')
                            label_str.append('<extra_id_' + str(t) + '>')
                            label_str.append('<extra_id_' + str(r + 101) + '>')
        else:
            for a in range(enum):
                for b in range(a + 1, enum):
                    for c in range(b + 1, enum):
                        for d in range(c + 1, enum):
                            if {a, b, c, d} in full_labels:
                                label_str.append('<extra_id_' + str(a) + '>')
                                label_str.append('<extra_id_' + str(b) + '>')
                                label_str.append('<extra_id_' + str(c) + '>')
                                label_str.append('<extra_id_' + str(d) + '>')
                                label_str.append('<extra_id_' + str(101) + '>')
        label_str += extra_labels
        label_str = ' '.join(label_str)
        labels = tokenizer(label_str, truncation=True, max_length=args.max_label_length).input_ids

        # fout.write(label_str + '\t' + str(len(labels)) + '\n')

        features.append({'title': input_doc['doc_id'],
                         'enum': enum,
                         'ref': label_str,
                         'input_ids': input_ids,
                         'labels': labels})
    # fout.close()
    ret['dev'] = features
    return ret


if __name__ == '__main__':
    # train_annotated: 35615 pos, 104446 win1 neg, 1163035 neg
    # dev: 11518 pos, 34417 win1 neg, 385272 neg
    logging.basicConfig(level=logging.INFO)
    import argparse
    from transformers import T5Tokenizer, BartTokenizer, LEDTokenizer
    # tokenizer = T5Tokenizer.from_pretrained('t5-large', extra_ids=150)
    # # dataset
    # args = {'data_dir': '../DocRED', 'max_input_length': 1024, 'max_label_length': 512}
    # save_dir = 'tmp_data_dir'
    # datasets = read_docred(args, ['train_annotated', 'dev'], save_dir,
    #                        tokenizer, refresh=True)

    tokenizer = LEDTokenizer.from_pretrained('allenai/led-large-16384', add_prefix_space=True,
                                             additional_special_tokens=[f"<extra_id_{i}>" for i in range(150)])
    # dataset
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data_dir = '../data/scirex/release_data'
    args.max_input_length = 4096
    args.max_label_length = 1024
    args.local_rank = -1
    args.cardinality = 4
    save_dir = 'tmp_data_dir'
    # datasets = read_scirex(args, ['test', 'train'], save_dir, tokenizer, refresh=True)
    datasets = read_scirex_test(args, '../data/scirex/pred-full-conll/test.log.jsonl', save_dir, tokenizer, refresh=True)
