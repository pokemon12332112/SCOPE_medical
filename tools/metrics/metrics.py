from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from .Radgraph import F1RadGraph
from .f1chexbert import F1CheXbert
import re
import numpy as np
import torch


def compute_nlg_scores(gts, res, args=None):
    gts = {i: [re.sub(' +', ' ', gt.replace(".", " ."))] for i, gt in enumerate(gts)}
    res = {i: [re.sub(' +', ' ', hpy.replace(".", " ."))] for i, hpy in enumerate(res)}
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), 'CIDer'),
    ]
    eval_res = {}
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


def compute_ce_scores(gts, res, args):

    f1chexbert = F1CheXbert(chexbert_checkpoint=args['chexbert_path'], model_checkpoint=args['bert_path'],
                            tokenizer_checkpoint=args['bert_path'])
    accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = f1chexbert(hyps=res, refs=gts)

    chexbert_5_micro_f1 = chexbert_5["micro avg"]
    chexbert_all_micro_f1 = chexbert_all["micro avg"]
    chexbert_5_macro_f1 = chexbert_5["macro avg"]
    chexbert_all_macro_f1 = chexbert_all["macro avg"]


    f1radgraph_partial = F1RadGraph(reward_level='partial', model_path=args['radgraph_path'])
    partial_mean_reward, reward_list, hypothesis_ann_lists, reference_ann_lists = f1radgraph_partial(hyps=res, refs=gts)


    metrics = {
 
        "F1-Radgraph-partial": partial_mean_reward,
        "chexbert_5_micro_f1": chexbert_5_micro_f1["f1-score"],
        "chexbert_5_macro_f1": chexbert_5_macro_f1["f1-score"],
        "chexbert_all_micro_p": chexbert_all_micro_f1['precision'],
        "chexbert_all_micro_r": chexbert_all_micro_f1['recall'],
        "chexbert_all_micro_f1": chexbert_all_micro_f1["f1-score"],
        "chexbert_all_macro_p": chexbert_all_macro_f1['precision'],
        "chexbert_all_macro_r": chexbert_all_macro_f1['recall'],
        "chexbert_all_macro_f1": chexbert_all_macro_f1["f1-score"],
    }

    return metrics


def compute_all_scores(gts, gens, args):

    ce_metrics = compute_ce_scores(gts, gens, args)


    nlg_metrics = compute_nlg_scores(gts, gens)
    ce_metrics.update(nlg_metrics)
    return ce_metrics


def compute_chexbert_scores(gts, gens, args):

    ce_metrics = compute_ce_scores(gts, gens, args)
    return ce_metrics


def compute_chexbert_details_scores(gts, res, args):
    f1chexbert = F1CheXbert(chexbert_checkpoint=args['chexbert_path'], model_checkpoint=args['bert_path'],
                            tokenizer_checkpoint=args['bert_path'])
    accuracy, accuracy_per_sample, chexbert_all, chexbert_5 = f1chexbert(hyps=res, refs=gts)

    del chexbert_all['weighted avg']
    del chexbert_all['samples avg']
    sample_num = chexbert_all['micro avg']['support']
    new_results = {}
    for key, value in chexbert_all.items():
        if 'avg' in key:
            new_results[key] = ['-', round(value['precision'], 3), round(value['recall'], 3), round(value['f1-score'], 3)]
        else:
            new_results[key] = [f"{round(value['support'] * 100 / sample_num, 1)} ({int(value['support'])})",
                                round(value['precision'], 3), round(value['recall'], 3), round(value['f1-score'], 3)]
    return new_results
