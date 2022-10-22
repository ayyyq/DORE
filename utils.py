import os
import json
import torch
from transformers import LogitsProcessor
from itertools import combinations


def collate_fn(batch):
    max_len = max([len(f['input_ids']) for f in batch])
    input_ids = [f['input_ids'] + [0] * (max_len - len(f['input_ids'])) for f in batch]
    attention_mask = [[1] * len(f['input_ids']) + [0] * (max_len - len(f['input_ids'])) for f in batch]

    max_len = max([len(f['labels']) for f in batch])
    labels = [f['labels'] + [-100] * (max_len - len(f['labels'])) for f in batch]

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    # LED
    # create 0 global_attention_mask lists
    global_attention_mask = torch.zeros_like(input_ids)
    # since above lists are references, the following line changes the 0 index for all samples
    global_attention_mask[:, 0] = 1
    return input_ids, attention_mask, labels, global_attention_mask


def update_embeddings(model, tokenizer, args):
    rel_value = []
    if 'docred' in args.data_dir.lower():
        jrel = json.load(open(os.path.join(args.data_dir, 'rel_info.json'), 'r'))
        for value in jrel.values():
            rel_value.append(value)
        rel_value.append('irrelevant')
    elif 'cdr' in args.data_dir.lower():
        rel_value.extend(['chemical-induced disease', 'irrelevant'])
    elif 'gda' in args.data_dir.lower():
        rel_value.extend(['gene-disease association', 'irrelevant'])
    elif 'scirex' in args.data_dir.lower():
        if args.end2end:
            map_available_entity_to_true = {"Material": "dataset", "Metric": "metric", "Task": "task",
                                            "Method": "model_name"}
            used_entities = list(map_available_entity_to_true.keys())
            for types in combinations(used_entities, args.cardinality):
                rel_typ = ' and '.join(types).lower()
                rel_value.append(rel_typ)
            rel_value.append('irrelevant')
        else:
            rel_value.extend(['relevant', 'irrelevant'])

    embed = model.get_input_embeddings().weight.data
    if args.model_type == 't5':
        for i_r, rel in enumerate(rel_value):
            rel_input_ids = tokenizer(rel).input_ids
            rel_embed = embed[rel_input_ids[:-1]].mean(dim=0)
            extra_idx = tokenizer('<extra_id_' + str(i_r) + '>').input_ids[0]
            embed[extra_idx] = rel_embed

        for i_e in range(50):
            idx = tokenizer(str(i_e)).input_ids[0]
            extra_idx = tokenizer('<extra_id_' + str(i_e + 101) + '>').input_ids[0]
            embed[extra_idx] = embed[idx]
    elif 'gda' in args.data_dir.lower() or 'scirex' in args.data_dir.lower():
        for i_r, rel in enumerate(rel_value):
            rel_input_ids = tokenizer(rel).input_ids
            rel_embed = embed[rel_input_ids[1:-1]].mean(dim=0)
            extra_idx = tokenizer('<extra_id_' + str(i_r + 101) + '>').input_ids[1]
            embed[extra_idx] = rel_embed

        for i_e in range(100):
            idx = tokenizer(str(i_e)).input_ids[1]
            extra_idx = tokenizer('<extra_id_' + str(i_e) + '>').input_ids[1]
            embed[extra_idx] = embed[idx]
    else:
        for i_r, rel in enumerate(rel_value):
            rel_input_ids = tokenizer(rel).input_ids
            rel_embed = embed[rel_input_ids[1:-1]].mean(dim=0)
            extra_idx = tokenizer('<extra_id_' + str(i_r) + '>').input_ids[1]
            embed[extra_idx] = rel_embed

        for i_e in range(50):
            idx = tokenizer(str(i_e)).input_ids[1]
            extra_idx = tokenizer('<extra_id_' + str(i_e + 101) + '>').input_ids[1]
            embed[extra_idx] = embed[idx]


class ConstrainedDecodingLogitsProcessor(LogitsProcessor):
    def __init__(self, args, bos_token_id, eos_token_id):
        super().__init__()
        self.args = args

        if not isinstance(eos_token_id, int) or eos_token_id < 0:
            raise ValueError(f"`eos_token_id` has to be a positive integer, but is {eos_token_id}")

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]

        if self.args.model_type == 't5':
            # input_ids[0] == decoder_start_token_id = 0
            # eos_token_id = 1, pad_token_id = 0
            # t5: (<extra_id_0>, 32149), (<extra_id_96>, 32053), (<extra_id_101>, 32048), (<extra_id_149>, 32000)

            # 1. <h, t, r>
            # 2. h_2 >= h_1
            if (cur_len - 1) % 3 == 0:
                # head entity
                scores[:, (self.eos_token_id + 1):32000] = -float("inf")
                scores[:, 32049:] = -float("inf")  # [32000:32048]
                if cur_len > 1:
                    for i in range(input_ids.shape[0]):
                        head_entity_start = input_ids[i, cur_len - 3].item()
                        scores[i, (head_entity_start + 1):] = -float("inf")
            elif (cur_len - 1) % 3 == 1:
                # tail entity
                scores[:, (self.eos_token_id + 1):32000] = -float("inf")
                scores[:, 32049:] = -float("inf")
            else:
                # relation
                scores[:, (self.eos_token_id + 1):32053] = -float("inf")  # [32053:32149]
        elif 'cdr' in self.args.data_dir.lower():
            # input_ids[0] = decoder_start_token_id = eos_token_id = 2
            # input_ids[1] = bos_token_id = 0
            # biobart: (<extra_id_0>, 50265), (<extra_id_101>, 50366), (<extra_id_126>, 50391), (<extra_id_149>, 50414)

            if cur_len >= 2:
                scores[:, self.bos_token_id] = -float("inf")  # not bos_token_id

                # 1. <h, t, r>
                # 2. h_2 >= h_1
                if (cur_len - 2) % 3 == 0:
                    # head entity
                    scores[:, (self.eos_token_id + 1):50366] = -float("inf")  # [50366:50391)
                    scores[:, 50391:] = -float("inf")
                    if cur_len > 2:
                        for i in range(input_ids.shape[0]):
                            head_entity_start = input_ids[i, cur_len - 3].item()
                            scores[i, (self.eos_token_id + 1):head_entity_start] = -float("inf")
                elif (cur_len - 2) % 3 == 1:
                    # tail entity
                    scores[:, (self.eos_token_id + 1):50391] = -float("inf")  # [50391:50415)
                else:
                    # relation
                    scores[:, (self.eos_token_id + 1):50265] = -float("inf")  # [50265, 50266]
                    scores[:, 50267:] = -float("inf")
        elif 'gda' in self.args.data_dir.lower():
            # input_ids[0] = decoder_start_token_id = eos_token_id = 2
            # input_ids[1] = bos_token_id = 0
            # biobart: (<extra_id_0>, 50265), (<extra_id_50>, 50315), (<extra_id_101>, 50366)

            if cur_len >= 2:
                scores[:, self.bos_token_id] = -float("inf")  # not bos_token_id

                # 1. <h, t, r>
                # 2. h_2 >= h_1
                if (cur_len - 2) % 3 == 0:
                    # head entity
                    scores[:, (self.eos_token_id + 1):50265] = -float("inf")  # [50265:50315)
                    scores[:, 50315:] = -float("inf")
                    if cur_len > 2:
                        for i in range(input_ids.shape[0]):
                            head_entity_start = input_ids[i, cur_len - 3].item()
                            scores[i, (self.eos_token_id + 1):head_entity_start] = -float("inf")
                elif (cur_len - 2) % 3 == 1:
                    # tail entity
                    scores[:, (self.eos_token_id + 1):50315] = -float("inf")  # [50315:50365)
                    scores[:, 50365:] = -float("inf")
                else:
                    # relation
                    scores[:, (self.eos_token_id + 1):50366] = -float("inf")  # [50366, 50367]
                    scores[:, 50368:] = -float("inf")
        elif 'scirex' in self.args.data_dir.lower() and self.args.cardinality == 2 and self.args.end2end:
            # input_ids[0] = decoder_start_token_id = eos_token_id = 2
            # input_ids[1] = bos_token_id = 0
            # led: (<extra_id_0>, 50265), (<extra_id_101>, 50366)

            if cur_len >= 2:
                scores[:, self.bos_token_id] = -float("inf")  # not bos_token_id

                # 1. <h, t, r>
                # 2. h2 >= h1, t > h
                if (cur_len - 2) % 3 == 0:
                    # head entity
                    scores[:, (self.eos_token_id + 1):50265] = -float("inf")  # [50265:50365)
                    scores[:, 50365:] = -float("inf")
                    if cur_len > 2:
                        for i in range(input_ids.shape[0]):
                            head_entity_start = input_ids[i, cur_len - 3].item()
                            scores[i, (self.eos_token_id + 1):head_entity_start] = -float("inf")
                elif (cur_len - 2) % 3 == 1:
                    # tail entity
                    scores[:, (self.eos_token_id + 1):50265] = -float("inf")  # [50265:50365)
                    scores[:, 50365:] = -float("inf")
                    for i in range(input_ids.shape[0]):
                        head_entity_start = input_ids[i, cur_len - 1].item() + 1  # the last token is head entity
                        scores[i, (self.eos_token_id + 1):head_entity_start] = -float("inf")
                else:
                    # relation
                    scores[:, (self.eos_token_id + 1):50366] = -float("inf")  # [50366:50373)
                    scores[:, 50373:] = -float("inf")
        elif 'scirex' in self.args.data_dir.lower() and self.args.cardinality == 4 and self.args.end2end:
            # input_ids[0] = decoder_start_token_id = eos_token_id = 2
            # input_ids[1] = bos_token_id = 0
            # led: (<extra_id_0>, 50265), (<extra_id_30>, 50295), (<extra_id_101>, 50366)

            if cur_len >= 2:
                scores[:, self.bos_token_id] = -float("inf")  # not bos_token_id

                # 1. <a, b, c, d, r>
                # 2. a2 >= a1
                if (cur_len - 2) % 5 == 0:
                    scores[:, (self.eos_token_id + 1):50265] = -float("inf")  # [50265:50365)
                    scores[:, 50365:] = -float("inf")
                    if cur_len > 2:
                        for i in range(input_ids.shape[0]):
                            head_entity_start = input_ids[i, cur_len - 5].item()
                            scores[i, (self.eos_token_id + 1):head_entity_start] = -float("inf")
                elif (cur_len - 2) % 5 != 4:
                    scores[:, (self.eos_token_id + 1):50265] = -float("inf")  # [50265:50365)
                    scores[:, 50365:] = -float("inf")
                    for i in range(input_ids.shape[0]):
                        head_entity_start = input_ids[i, cur_len - 1].item() + 1  # the last token is head entity
                        scores[i, (self.eos_token_id + 1):head_entity_start] = -float("inf")
                else:
                    # relation
                    scores[:, (self.eos_token_id + 1):50366] = -float("inf")  # [50366, 50367]
                    scores[:, 50368:] = -float("inf")
        elif 'scirex' in self.args.data_dir.lower() and self.args.cardinality == 2:
            # input_ids[0] = decoder_start_token_id = eos_token_id = 2
            # input_ids[1] = bos_token_id = 0
            # led: (<extra_id_0>, 50265), (<extra_id_30>, 50295), (<extra_id_101>, 50366)

            if cur_len >= 2:
                scores[:, self.bos_token_id] = -float("inf")  # not bos_token_id

                # 1. <h, t, r>
                # 2. h2 >= h1, t > h
                if (cur_len - 2) % 3 == 0:
                    # head entity
                    scores[:, (self.eos_token_id + 1):50265] = -float("inf")  # [50265:50295)
                    scores[:, 50295:] = -float("inf")
                    if cur_len > 2:
                        for i in range(input_ids.shape[0]):
                            head_entity_start = input_ids[i, cur_len - 3].item()
                            scores[i, (self.eos_token_id + 1):head_entity_start] = -float("inf")
                elif (cur_len - 2) % 3 == 1:
                    # tail entity
                    scores[:, (self.eos_token_id + 1):50275] = -float("inf")  # [50275:50365)
                    scores[:, 50365:] = -float("inf")
                    for i in range(input_ids.shape[0]):
                        head_entity_start = input_ids[i, cur_len - 1].item()  # the last token is head entity
                        head_entity_start = ((head_entity_start - 50265) // 10 + 1) * 10 + 50265
                        scores[i, (self.eos_token_id + 1):head_entity_start] = -float("inf")
                else:
                    # relation
                    scores[:, (self.eos_token_id + 1):50366] = -float("inf")  # [50366, 50367]
                    scores[:, 50368:] = -float("inf")
        elif 'scirex' in self.args.data_dir.lower() and self.args.cardinality == 4:
            # input_ids[0] = decoder_start_token_id = eos_token_id = 2
            # input_ids[1] = bos_token_id = 0
            # led: (<extra_id_0>, 50265), (<extra_id_30>, 50295), (<extra_id_101>, 50366)

            if cur_len >= 2:
                scores[:, self.bos_token_id] = -float("inf")  # not bos_token_id

                # 1. <a, b, c, d, r>
                # 2. a2 >= a1
                if (cur_len - 2) % 5 == 0:
                    scores[:, (self.eos_token_id + 1):50265] = -float("inf")  # [50265:50275)
                    scores[:, 50275:] = -float("inf")
                    if cur_len > 2:
                        for i in range(input_ids.shape[0]):
                            head_entity_start = input_ids[i, cur_len - 5].item()
                            scores[i, (self.eos_token_id + 1):head_entity_start] = -float("inf")
                elif (cur_len - 2) % 5 == 1:
                    scores[:, (self.eos_token_id + 1):50275] = -float("inf")  # [50275:50285)
                    scores[:, 50285:] = -float("inf")
                elif (cur_len - 2) % 5 == 2:
                    scores[:, (self.eos_token_id + 1):50285] = -float("inf")  # [50285:50295)
                    scores[:, 50295:] = -float("inf")
                elif (cur_len - 2) % 5 == 3:
                    scores[:, (self.eos_token_id + 1):50295] = -float("inf")  # [50295:50365)
                    scores[:, 50365:] = -float("inf")
                else:
                    # relation
                    scores[:, (self.eos_token_id + 1):50366] = -float("inf")  # [50366, 50367]
                    scores[:, 50368:] = -float("inf")
        else:
            # FOR BART or LED on DocRED
            # input_ids[0] = decoder_start_token_id = eos_token_id = 2
            # input_ids[1] = bos_token_id = 0
            # led: (<extra_id_0>, 50265), (<extra_id_96>, 50361), (<extra_id_101>, 50366), (<extra_id_149>, 50414)

            if cur_len >= 2:
                scores[:, self.bos_token_id] = -float("inf")  # not bos_token_id

                # 1. <h, t, r>
                # 2. h_2 >= h_1
                if (cur_len - 2) % 3 == 0:
                    # head entity
                    scores[:, (self.eos_token_id + 1):50366] = -float("inf")  # [50366:50415)
                    if cur_len > 2:
                        for i in range(input_ids.shape[0]):
                            if input_ids[i, cur_len - 3].item() >= 50366:  # last head entity
                                head_entity_start = input_ids[i, cur_len - 3].item()
                                scores[i, (self.eos_token_id + 1):head_entity_start] = -float("inf")
                elif (cur_len - 2) % 3 == 1:
                    # tail entity
                    scores[:, (self.eos_token_id + 1):50366] = -float("inf")
                else:
                    # relation
                    scores[:, (self.eos_token_id + 1):50265] = -float("inf")  # [50265:50362)
                    scores[:, 50362:] = -float("inf")
        return scores
