import argparse
from pathlib import Path

from utils import add_jobid


def get_argparser():
    desc = 'TODO'
    a = argparse.ArgumentParser(description=desc)
    arg = a.add_argument
    arg('command', type=str)
    arg('--exp-name', type=str, default='dev')
    arg('--backend-store-uri', type=str, default='sqlite:///mlflow.db')
    arg('--no-train', action='store_true')
    arg('--no-cache', action='store_true')
    arg('--dev', action='store_true')
    arg('--outdir', type=Path, default='out')
    arg('--datadir', type=Path, default='data')
    arg('--rundir', type=Path)
    arg('--checkpointer-state-file', type=Path)
    arg('--trainer-state-file', type=Path)
    arg('--runid', type=str)
    arg('--mlflow-runid', type=str)
    arg('--jobid', type=str)
    arg('--inspect', action='store_true')
    arg('--force-new-run', action='store_true')
    arg('--force-finish-exp', type=int)
    arg('--end-run', action='store_true')
    arg('--random-seed', type=int, default=1)
    arg('--device', type=str, default='cuda:0')
    arg('--distributed', action='store_true')
    arg('--dist-backend', type=str, default='nccl')
    arg('--dist-master-addr', type=str, default='127.0.0.1')
    arg('--dist-master-port', type=str, default='29505')
    arg("--local-rank", type=int, default=0)
    arg("--dist-init-method", type=str, default="env://")
    arg('--stdout-dir', type=Path, default='~/uge')

    arg('--dataset', type=str, default='wikidatarelationstatements')
    arg('--top-n', type=int, required=True)
    arg('--n-facts', type=int, required=True)
    arg('--entity-repr', type=str, required=True)

    arg('--wikidata-dir', type=Path, default='data/wikidata')
    arg('--entities-file-tpl', type=str, default='instances.top{top_n}.jl')
    arg('--labels-en-file-tpl', type=str, default='label_en.top{top_n}.tsv')
    arg('--target-arg', type=str, default='o')
    arg('--lang', type=str, default='en')
    arg('--ambiguous-subj-pred', action='store_true')
    arg('--ambiguous-pred-obj', action='store_true')
    arg('--n-neg-samples', type=int, default=0)
    arg('--no-shuffle', action='store_true')
    arg('--split-dataset', action='store_true')
    arg('--train-split-ratio', type=float, default=0.7)

    arg("--plot-kb-emb", action="store_true")
    arg("--plot-reprs", action="store_true")
    arg('--results-dir', type=Path, default='out/results')
    arg('--no-print-examples', action='store_true')
    arg('--no-checkpoints', action='store_true')
    arg('--transformer-model', type=str, default='roberta-base')
    arg('--max-seq-len', type=int, default=60)
    arg('--clip-to-max-seq-len', action='store_true')
    arg('--vary-n-params-memorization-threshold', type=float, default=0.95)

    arg('--architecture', type=str, choices=['lstm', 'transformer'], required=True)
    arg('--tokenizer', type=str, default='roberta-base')
    arg('--model-file', type=Path)
    arg('--no-fp16', action='store_true')
    arg('--no-load-state', action='store_true')
    arg('--no-load-optim', action='store_true')

    arg('--n-layers', type=int)
    arg('--n-hidden', type=int)

    arg('--rnn-type', type=str, default='LSTM')
    arg('--rnn-dropout', type=float, default=0.0)
    arg('--rnn-emb-dim', type=int, default=100)
    arg('--rnn-emb-random-init', action='store_true')
    arg('--freeze-emb', action='store_true')
    arg('--use-attention', action='store_true')
    arg('--no-attention', action='store_true')
    arg('--trf-pool', type=str, default='mask')
    arg('--softmax-size',
        type=int, default=128, help='size of the softmax layer')
    arg(
        '--pooling', type=str, choices=['mask', 'mean'], default='mask')
    arg('--encoder-pretrained-emb-file', type=Path)

    arg('--code-predictor', action='store_true')
    arg('--kb-emb-random-init', action='store_true')
    arg('--kb-emb-dim', type=int, default=64)
    arg('--kb-emb-index-update-interval', type=int, default=10)
    arg('--kb-emb-file', type=Path)
    arg('--kb-emb-trainable', action='store_true')
    arg('--kb-emb-non-uniform', action='store_true')
    arg('--kb-emb-normalized', action='store_true')
    arg(
        '--kb-emb-code-file-tpl', type=str,
        default='binary_code/kdtree.binary_code.top{top}.npy')
    arg('--predictor-dev', type=str)

    arg('--max-ctx-subw-len', type=int, default=64)
    arg('--max-target-subw-len', type=int, default=48)
    arg('--repr-dim', type=int)

    arg('--batch-size', type=int, default=128)
    arg('--eval-batch-size', type=int, default=32)
    arg('--optim', type=str, default='adam')
    arg('--max-grad-norm', type=float, default=1.0)
    arg('--predictor-lr', type=float, default=5e-5)
    arg("--lr-scheduler", type=str, default='plateau')
    arg("--lr-scheduler-patience", type=int, default=20)
    arg('--lr', type=float, default=5e-5)
    arg('--lr-metric-name', type=str, default='loss')
    arg('--lr-metric-optimum', type=str, default='min')
    arg('--warmup-steps', type=int, default=100)
    arg('--n-train-steps', type=int, default=100000)
    arg('--momentum', type=float, default=0.0)
    arg('--weight-decay', type=float, default=0.0)
    arg('--early-stopping', type=int, default=20)
    arg('--early-stopping-burnin', type=int, default=20)
    arg('--eval-every', type=int, default=1)
    arg('--first-eval-epoch', type=int, default=1)
    arg('--first-checkpoint-after', type=int, default=1)
    arg('--max-epochs', type=int, default=1000)
    arg('--max-eval-inst', type=int, default=1000)
    arg('--max-train-inst', type=int)
    arg('--max-test-inst', type=int, default=1000)
    arg('--write-predictions', action='store_true')

    arg('--paraphrase-mode', type=str, choices=['train', 'finetune'])
    arg('--paraphrase-id', type=int)
    arg('--n-finetune-insts', type=int, nargs='+',
        default=[0, 10, 20, 50, 100, 200, 500])
    arg('--paraphrase-ids', type=int, nargs='+', default=list(range(1, 17)))
    arg('--paraphrase-sim-outfile', type=str,
        default='out/paraphrase-{i}.sims.json')
    arg('--paraphrase-dist-outfile', type=str,
        default='out/paraphrase-{i}.dist.json')
    arg('--memorization-threshold', type=float, default=0.99)

    return a


def set_kb_emb_pre(args):
    kb_emb_filename = {
        (1000, 32): 'entity.top1000.description_labels_quantities_relations_tags.32d.6dbea47e8aba6332af72e4a9c5ae948a.pth',
        (1000, 64): 'entity.top1000.description_labels_quantities_relations_tags.64d.7e2d9e6bd85ba02eb9a3ee68da5d6f67.pth',
        (1000, 128): 'entity.top1000.description_labels_quantities_relations_tags.128d.152c7b0ecea7e80ce5a304f332a1eb89.pth',
        (1000, 192): 'entity.top1000.description_labels_quantities_relations_tags.192d.11f5777dc6a526cf8a01bf94c3a86e81.pth',
        (1000, 256): 'entity.top1000.description_labels_quantities_relations_tags.256d.a1335031d75caaa7185329a87199a3ba.pth',

        (10000, 32): 'entity.top10000.description_labels_quantities_relations_tags.32d.2982e9997ad92383f0af602cd20eaff3.pth',
        (10000, 64): 'entity.top10000.description_labels_quantities_relations_tags.64d.8a0b92b9604cbf919eebe139e0727c36.pth',
        (10000, 128): 'entity.top10000.description_labels_quantities_relations_tags.128d.65e264453a942c7dadeda28ae835c389.pth',
        (10000, 192): 'entity.top10000.description_labels_quantities_relations_tags.192d.d5a7db11ea84f05d205b1ee422f6ef9f.pth',
        (10000, 256): 'entity.top10000.description_labels_quantities_relations_tags.256d.ad0396907977a94de586b4002b8cac41.pth',

        (100000, 32): 'entity.top100000.description_labels_quantities_relations_tags.32d.6fb40476861515c360c21af324779684.pth',
        (100000, 64): 'entity.top100000.description_labels_quantities_relations_tags.64d.e77f3de112d84f359adec9931c610c78.pth',
        (100000, 128): 'entity.top100000.description_labels_quantities_relations_tags.128d.9f4e67511a94b0990dda90fec669c752.pth',
        (100000, 192): 'entity.top100000.description_labels_quantities_relations_tags.192d.f0f24905a3f080a560e68f3ac5783875.pth',
        (100000, 256): 'entity.top100000.description_labels_quantities_relations_tags.256d.0b00ea0651aefc63adec3bcc6ba82ce6.pth',

        (1000000, 32): 'entity.top1000000.description_labels_quantities_relations_tags.32d.2fde3e41c210c971ef1f0cfa56aa520f.pth',
        (1000000, 64): 'entity.top1000000.description_labels_quantities_relations_tags.64d.f08e506976af2412dfc5ae86d4e6be1b.pth',
        (1000000, 128): 'entity.top1000000.description_labels_quantities_relations_tags.128d.a160954891ca2cb19323a3e47903eed0.pth',
        (1000000, 192): 'entity.top1000000.description_labels_quantities_relations_tags.192d.5e19395cea3bf6bfabde9755b53c5663.pth',
        (1000000, 256): 'entity.top1000000.description_labels_quantities_relations_tags.256d.64c33a5833c47f0c9fc965557b04ab5f.pth',

        (6000000, 64): 'entity.top6000000.description_labels_quantities_relations_tags.64d.0e0ce161c1f88176456535342adadc3c.pth',
        (6000000, 128): 'entity.top6000000.description_labels_quantities_relations_tags.128d.e469569ab2034abdb7e803417c087330.pth',
        (6000000, 192): 'entity.top6000000.description_labels_quantities_relations_tags.192d.1fdf75c9f4a0720e212d8d038a3e5109.pth',
        (6000000, 256): 'entity.top6000000.description_labels_quantities_relations_tags.256d.add2c454513b01d11ad3d15230013d14.pth',
    }[(args.top_n, args.kb_emb_dim)]
    if not args.kb_emb_non_uniform:
        suffix = 'uniform_spherical.pth'
        kb_emb_filename = kb_emb_filename[:-len('pth')] + suffix
    args.kb_emb_file = Path('emb') / kb_emb_filename


def set_kb_emb(args):
    if args.kb_emb_random_init:
        return
    set_kb_emb_pre(args)


def entity_repr_to_model_suffix(args):
    return {
        'embedding': 'continuous',
        'symbol': 'symbol'}[args.entity_repr]


def get_conf():
    a = get_argparser()
    args = a.parse_args()
    add_jobid(args)
    args.use_attention = not args.no_attention
    set_kb_emb(args)
    if args.architecture == 'lstm':
        assert args.n_layers
        assert args.n_hidden
    return args
