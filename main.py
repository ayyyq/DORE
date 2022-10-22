import yaml
import argparse
import tqdm
import os
import json
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from apex import amp
from transformers import T5Tokenizer, LEDTokenizer, BartTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import LogitsProcessorList
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from parse import parse_args
from data import read_docred, read_cdr, read_gda, read_scirex_using_gold_inputs, read_scirex, read_scirex_test
from utils import collate_fn, update_embeddings, ConstrainedDecodingLogitsProcessor
from evaluate import eval_docred, eval_f1

import logging
import fitlog


def save_model(model, model_path):
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].cpu()
    torch.save(state_dict, model_path)


def load_model(model, model_path):
    states = torch.load(model_path)
    model.load_state_dict(states)


def test(dataset, dataloader, model, tokenizer, args, result_path, num_beams=1, do_test=False):
    total_pred = []
    with tqdm.tqdm(dataloader) as tqb:
        for batch in tqb:
            model.eval()

            bos_token_id = tokenizer.bos_token_id if args.model_type != 't5' else -1
            logits_processor = LogitsProcessorList(
                [
                    ConstrainedDecodingLogitsProcessor(args, bos_token_id, tokenizer.eos_token_id),
                ]
            )

            inputs = {'input_ids': batch[0].to(model.device),
                      'attention_mask': batch[1].to(model.device),
                      'max_length': args.max_length,
                      'min_length': args.min_length,
                      'num_beams': num_beams,
                      'length_penalty': args.length_penalty,
                      'logits_processor': logits_processor}
            if args.model_type == 'led':
                inputs['global_attention_mask'] = batch[3].to(model.device)
            if args.model_type == 'bart' or args.model_type == 'led':
                inputs['forced_bos_token_id'] = tokenizer.bos_token_id
            with torch.no_grad():
                outputs = model.generate(**inputs)

            total_pred.extend(tokenizer.batch_decode(outputs))

    total_title = [f['title'] for f in dataset]
    total_enum = [f['enum'] for f in dataset]
    total_ref = [f['ref'] for f in dataset]
    total_filter_rel = []
    for f in dataset:
        if 'filter_rel' not in f:
            break
        total_filter_rel.append(f['filter_rel'])

    p, r, f1, ign_f1 = 0., 0., 0., 0.
    if 'docred' in args.data_dir.lower():
        id2rel = []  # rel_key
        jrel = json.load(open(os.path.join(args.data_dir, 'rel_info.json'), 'r'))
        for key in jrel.keys():
            id2rel.append(key)

        if not do_test:
            p, r, _ = eval_f1(args, total_pred, total_ref, total_title, total_enum, total_filter_rel, 96, result_path)
        f1, ign_f1 = eval_docred(args, total_pred, total_title, total_enum, id2rel, result_path, do_test)
    elif 'cdr' in args.data_dir.lower():
        p, r, f1 = eval_f1(args, total_pred, total_ref, total_title, total_enum, total_filter_rel, 1, result_path)
        ign_f1 = f1
    elif 'gda' in args.data_dir.lower() or ('scirex' in args.data_dir.lower() and (not args.end2end or args.cardinality == 4)):
        p, r, f1 = eval_f1(args, total_pred, total_ref, total_title, total_enum, total_filter_rel, 102, result_path)
        ign_f1 = f1
    elif 'scirex' in args.data_dir.lower() and args.end2end:
        p, r, f1 = eval_f1(args, total_pred, total_ref, total_title, total_enum, total_filter_rel, 107, result_path)
        ign_f1 = f1
    return p, r, f1, ign_f1


def eval_loss(dataloader, model, args):
    dev_losses = []
    with tqdm.tqdm(dataloader) as tqb:
        for batch in tqb:
            model.eval()

            inputs = {'input_ids': batch[0].to(model.device),
                      'attention_mask': batch[1].to(model.device),
                      'labels': batch[2].to(model.device)}
            if args.model_type == 'led':
                inputs['global_attention_mask'] = batch[3].to(model.device)
            with torch.no_grad():
                dev_loss = model(**inputs)[0]
            dev_losses.append(dev_loss.item())
    return np.mean(dev_losses)


def train(args, datasets, model, tokenizer, model_dir, result_save_dir):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(datasets['train']) if args.local_rank == -1 else DistributedSampler(datasets['train'])
    train_dataloader = DataLoader(datasets['train'], sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    dev_dataloader = DataLoader(datasets['dev'], batch_size=args.eval_batch_size, shuffle=False,
                                collate_fn=collate_fn)
    if 'test' in datasets:
        test_dataloader = DataLoader(datasets['test'], batch_size=args.eval_batch_size, shuffle=False,
                                     collate_fn=collate_fn)

    total_steps = int(len(train_dataloader) * args.num_epochs // args.accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    global_step = 0
    max_ign_f1 = 0
    best_step = 0
    best_epoch = 0
    losses = []
    start_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
    logging.info(start_time)
    best_model_path = os.path.join(model_dir, 'best_f1_model_' + str(start_time))
    model.zero_grad()

    for epoch in range(args.num_epochs):
        with tqdm.tqdm(train_dataloader, disable=args.local_rank not in [-1, 0]) as tqb:
            tqb.write('Epoch: ' + str(epoch))
            for step, batch in enumerate(tqb):
                model.train()

                inputs = {'input_ids': batch[0].to(model.device),
                          'attention_mask': batch[1].to(model.device),
                          'labels': batch[2].to(model.device)}
                if args.model_type == 'led':
                    inputs['global_attention_mask'] = batch[3].to(model.device)
                loss = model(**inputs)[0]
                if args.accumulation_steps > 1:
                    loss /= args.accumulation_steps

                if args.fp16:
                    # Scales loss.
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (step + 1) % args.accumulation_steps == 0:
                    # Gradient clipping
                    if args.max_grad_norm > 0:
                        if args.fp16:
                            nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    # Optimizer step
                    optimizer.step()
                    if args.warmup_ratio > 0:
                        scheduler.step()
                    model.zero_grad()

                losses.append(loss.item())
                tqb.set_postfix({'loss': np.mean(losses)})
                global_step += 1

                # Only evaluate when single GPU otherwise metrics may not average well
                if args.local_rank in [-1, 0] and args.eval_steps != -1 and global_step % args.eval_steps == 0:
                    # dev_loss = eval_loss(dev_dataloader, model, args)
                    # tqb.write('Epoch: ' + str(epoch) + ' Step: ' + str(global_step))
                    # tqb.write('Dev Loss ' + str(dev_loss))

                    result_path = os.path.join(result_save_dir,
                                               '_'.join(['result', start_time, str(global_step)]) + '.json')
                    p, r, f1, ign_f1 = test(datasets['dev'], dev_dataloader, model, tokenizer, args, result_path)
                    tqb.write('Epoch: ' + str(epoch) + ' Step: ' + str(global_step))
                    tqb.write('p: ' + str(p) + ' r: ' + str(r) + ' f1: ' + str(f1) + ' ign_f1: ' + str(ign_f1))
                    fitlog.add_metric({'dev': {'f1': f1, 'ign_f1': ign_f1}}, step=global_step, epoch=epoch)
                    if ign_f1 > max_ign_f1:
                        save_model(model, best_model_path)
                        max_ign_f1 = ign_f1
                        best_step = global_step
                        best_epoch = epoch
                        fitlog.add_best_metric({'dev': {'f1': f1, 'ign_f1': ign_f1,
                                                        'step': best_step, 'epoch': best_epoch}})

        # logging.info('Epoch ' + str(i))

        if args.local_rank in [-1, 0] and (args.eval_steps == -1 or global_step % args.eval_steps != 0):
            # each epoch
            # dev_loss = eval_loss(dev_dataloader, model, args)
            # tqb.write('Epoch: ' + str(epoch) + ' Step: ' + str(global_step))
            # tqb.write('Dev Loss ' + str(dev_loss))

            if args.eval_steps == -1:
                model_path = os.path.join(model_dir, '_'.join(['model', start_time, str(epoch)]))
                result_path = os.path.join(result_save_dir,
                                           '_'.join(['result', start_time, str(epoch)]) + '.json')
            else:
                model_path = os.path.join(model_dir, '_'.join(['model', start_time, str(global_step)]))
                result_path = os.path.join(result_save_dir,
                                           '_'.join(['result', start_time, str(global_step)]) + '.json')
            if args.save_model_per_batch:
                save_model(model, model_path)

            p, r, f1, ign_f1 = test(datasets['dev'], dev_dataloader, model, tokenizer, args, result_path)
            tqb.write('Epoch: ' + str(epoch) + ' Step: ' + str(global_step))
            tqb.write('p: ' + str(p) + ' r: ' + str(r) + ' f1: ' + str(f1) + ' ign_f1: ' + str(ign_f1))
            fitlog.add_metric({'dev': {'f1': f1, 'ign_f1': ign_f1}}, step=global_step, epoch=epoch)
            if ign_f1 > max_ign_f1:
                save_model(model, best_model_path)
                max_ign_f1 = ign_f1
                best_step = global_step
                best_epoch = epoch
                fitlog.add_best_metric({'dev': {'f1': f1, 'ign_f1': ign_f1, 'step': best_step, 'epoch': best_epoch}})

    if args.local_rank in [-1, 0]:
        if args.eval_steps == -1:
            model_path = os.path.join(model_dir, '_'.join(['model', start_time, str(args.num_epochs - 1)]))
        else:
            model_path = os.path.join(model_dir, '_'.join(['model', start_time, str(global_step)]))
        save_model(model, model_path)
        logging.info('Epoch: {:d} Step: {:d} max_ign_f1: {:.4f}'.format(best_epoch, best_step, max_ign_f1))

        if 'annotated' in args.train_data_type or 'docred' not in args.data_dir.lower():
            load_model(model, best_model_path)

            if args.fp16:
                model = amp.initialize(model, opt_level=args.fp16_opt_level)

            result_path = os.path.join(result_save_dir, '_'.join(['result', start_time, 'dev']) + '.json')
            p, r, f1, ign_f1 = test(datasets['dev'], dev_dataloader, model, tokenizer,
                                    args, result_path, num_beams=args.num_beams)
            logging.info('Best Epoch: {:d} Step: {:d} p: {:.4f} r: {:.4f} '
                         'f1: {:.4f} ign_f1: {:.4f}'.format(best_epoch, best_step, p, r, f1, ign_f1))
            fitlog.add_best_metric({'dev': {'f1': f1, 'ign_f1': ign_f1, 'step': best_step, 'epoch': best_epoch}})
            fitlog.finish()

            if 'docred' in args.data_dir.lower():
                result_path = os.path.join(result_save_dir, '_'.join(['result', start_time, 'test']) + '.json')
                test(datasets['test'], test_dataloader, model, tokenizer, args, result_path,
                     num_beams=args.num_beams, do_test=True)


def main():
    args = parse_args()

    # Set CUDA, GPU & distributed training
    if args.local_rank == -1:
        assert torch.cuda.is_available()
        device = torch.device('cuda')
        args.n_gpu = 1 if args.model_type == 't5' else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    top_save_dir = os.path.join('outputs', args.task_name)
    save_dir = 'output'
    if args.lr != 1e-4:
        save_dir += '_lr' + str(args.lr)
    if args.weight_decay != 0.01:
        save_dir += '_decay' + str(args.weight_decay)
    if args.warmup_ratio != 0:
        save_dir += '_warm' + str(args.warmup_ratio)
    if args.max_grad_norm != 1.0:
        save_dir += '_norm' + str(args.max_grad_norm)
    if args.n_gpu * args.per_gpu_train_batch_size * args.accumulation_steps != 4:
        if args.accumulation_steps < 0:
            save_dir += '_bsz' + str(args.per_gpu_train_batch_size)
        else:
            save_dir += '_bsz' + str(args.per_gpu_train_batch_size * args.accumulation_steps)
    if args.max_length != 300:
        save_dir += '_maxl' + str(args.max_length)
    if args.min_length != 10:
        save_dir += '_minl' + str(args.min_length)
    if args.num_beams != 4:
        save_dir += '_nbeams' + str(args.num_beams)
    if args.length_penalty != 1.0:
        save_dir += '_pen' + str(args.length_penalty)
    args.save_dir = os.path.join(top_save_dir, save_dir)
    model_dir = os.path.join(args.save_dir, args.model_dir)
    result_save_dir = os.path.join(args.save_dir, args.result_dir)

    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logging.info('Start Logging')
    logging.info(args)
    # fitlog.debug()
    if args.local_rank > 0 or args.do_test:
        fitlog.debug()
    fitlog.commit(__file__)
    fitlog.set_log_dir('logs')
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)
    fitlog.set_rng_seed()
    #####hyper
    #####hyper

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir, exist_ok=True)

    if args.model_type == 't5':
        tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path, extra_ids=150)
    elif args.model_type == 'bart':
        tokenizer = BartTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                                  additional_special_tokens=[f"<extra_id_{i}>" for i in range(150)],
                                                  add_prefix_space=True)
    elif args.model_type == 'led':
        tokenizer = LEDTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                                 additional_special_tokens=[f"<extra_id_{i}>" for i in range(150)],
                                                 add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

    # load and cache datasets
    if 'cdr' in args.data_dir.lower():
        datasets = read_cdr(args, [args.train_data_type, 'test'], top_save_dir, tokenizer, refresh=args.refresh)
    elif 'gda' in args.data_dir.lower():
        datasets = read_gda(args, [args.train_data_type, 'test'], top_save_dir, tokenizer, refresh=args.refresh)
    elif 'scirex' in args.data_dir.lower():
        if args.end2end:
            if args.do_test:
                datasets = read_scirex_test(args, '../data/scirex/pred-full-conll/test.log.jsonl', args.save_dir, tokenizer, refresh=args.refresh)
            else:
                datasets = read_scirex(args, [args.train_data_type, 'test'], top_save_dir, tokenizer, refresh=args.refresh)
        else:
            datasets = read_scirex_using_gold_inputs(args, [args.train_data_type, 'test'], top_save_dir, tokenizer, refresh=args.refresh)
    else:
        datasets = read_docred(args, [args.train_data_type, 'dev', 'test'], top_save_dir, tokenizer, refresh=args.refresh)


    if args.model_type == 'led':
        model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name_or_path,
                                                      gradient_checkpointing=True, use_cache=False)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    update_embeddings(model, tokenizer, args)
    if args.load_model_path != "":
        load_model(model, args.load_model_path)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.model_type == 't5':
        assert args.local_rank == -1
        device_map = {0: [0, 1, 2],
                      1: [3, 4, 5, 6, 7, 8, 9],
                      2: [10, 11, 12, 13, 14, 15, 16],
                      3: [17, 18, 19, 20, 21, 22, 23]}
        model.parallelize(device_map)
    else:
        model.to(args.device)

    if args.do_test:
        if args.fp16:
            model = amp.initialize(model, opt_level=args.fp16_opt_level)

        assert args.local_rank == -1
        dev_dataloader = DataLoader(datasets['dev'], batch_size=args.eval_batch_size, shuffle=False,
                                    collate_fn=collate_fn)
        if 'test' in datasets:
            test_dataloader = DataLoader(datasets['test'], batch_size=args.eval_batch_size, shuffle=False,
                                         collate_fn=collate_fn)

        start_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f'))
        logging.info(start_time)
        result_path = os.path.join(result_save_dir, '_'.join(['result', start_time, 'dev']) + '.json')
        p, r, f1, ign_f1 = test(datasets['dev'], dev_dataloader, model, tokenizer,
                                args, result_path, num_beams=args.num_beams)
        logging.info('p: {:.4f} r: {:.4f} f1: {:.4f} ign_f1: {:.4f}'.format(p, r, f1, ign_f1))

        if 'docred' in args.data_dir.lower():
            result_path = os.path.join(result_save_dir, '_'.join(['result', start_time, 'test']) + '.json')
            test(datasets['test'], test_dataloader, model, tokenizer, args, result_path,
                 num_beams=args.num_beams, do_test=True)
        return

    train(args, datasets, model, tokenizer, model_dir, result_save_dir)


if __name__ == '__main__':
    main()
