import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='bart-win1', type=str)
    parser.add_argument('--cardinality', default=2, choices=[2, 4], type=int)
    parser.add_argument('--end2end', action='store_true')
    parser.add_argument('--model_type', default='bart', choices=['t5', 'led', 'bart'], type=str)
    parser.add_argument('--pretrained_model_name_or_path', default='facebook/bart-large', type=str)
    parser.add_argument('--data_dir', default='../DocRED', type=str)
    parser.add_argument('--train_data_type', default='train_annotated', type=str)
    parser.add_argument('--model_dir', default='checkpoint', type=str)
    parser.add_argument('--load_model_path', default='', type=str)
    parser.add_argument('--result_dir', default='result', type=str)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--max_input_length', default=1024, type=int)
    parser.add_argument('--max_label_length', default=512, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)  # 3e-5 for Bart and 1e-4 for T5
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_ratio', default=0, type=float)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--num_epochs', default=40, type=int)
    parser.add_argument('--eval_steps', default=-1, type=int)
    parser.add_argument('--save_model_per_batch', action='store_true')
    parser.add_argument('--per_gpu_train_batch_size', default=2, type=int)
    parser.add_argument('--accumulation_steps', default=2, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument('--max_length', default=300, type=int)
    parser.add_argument('--min_length', default=10, type=int)
    parser.add_argument('--num_beams', default=4, type=int)
    parser.add_argument('--length_penalty', default=1.0, type=float)

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', default='O1', type=str)

    parser.add_argument('--local_rank', default=-1, type=int)

    args = parser.parse_args()
    return args
