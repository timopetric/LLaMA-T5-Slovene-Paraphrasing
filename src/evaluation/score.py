import os
import sys
import time
import pathlib
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu
from collections import defaultdict
from transformers import AutoTokenizer

from utils import (
    get_model,
    get_tokenizer,
    get_idf_dict,
    bert_cos_score_idf,
    diverse
)


__all__ = ["score"]


def score(
    cands,
    refs,
    model_type="EMBEDDIA/sloberta",
    num_layers=9,
    idf=False,
    device=None,
    batch_size=64,
    nthreads=4,
    all_layers=False,
    use_fast_tokenizer=True,
    diversity_factor=0.05,
    use_bleu=False
):
    """
    BERTScore metric.

    Args:
        - :param: `cands` (list of str): candidate sentences
        - :param: `refs` (list of str or list of list of str): reference sentences
        - :param: `model_type` (str): bert specification, default using the suggested
                  model for the target langauge; has to specify at least one of
                  `model_type` or `lang`
        - :param: `num_layers` (int): the layer of representation to use.
                  default using the number of layer tuned on WMT16 correlation data
        - :param: `idf` (bool or dict): use idf weighting, can also be a precomputed idf_dict
        - :param: `device` (str): on which the contextual embedding model will be allocated on.
                  If this argument is None, the model lives on cuda:0 if cuda is available.
        - :param: `nthreads` (int): number of threads
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `use_fast_tokenizer` (bool): `use_fast` parameter passed to HF tokenizer
        - :param: `diversity_factor` (float): How much diversity contributes to final paraphrase score
        - :param: `use_bleu` (bool): use bleu score instead of edit distance
                    for penalizing sentences that are too similar

    Return:
        - :param: `(F1)`: each is of shape (N); N = number of input
                  candidate reference pairs.
    """
    assert len(cands) == len(refs), "Different number of candidates and references"


    ref_group_boundaries = None
    if not isinstance(refs[0], str):
        ref_group_boundaries = []
        ori_cands, ori_refs = cands, refs
        cands, refs = [], []
        count = 0
        for cand, ref_group in zip(ori_cands, ori_refs):
            cands += [cand] * len(ref_group)
            refs += ref_group
            ref_group_boundaries.append((count, count + len(ref_group)))
            count += len(ref_group)

    tokenizer = get_tokenizer(model_type, use_fast_tokenizer)
    model = get_model(model_type, num_layers, all_layers)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if not idf:
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
    elif isinstance(idf, dict):
        idf_dict = idf
    else:
        idf_dict = get_idf_dict(refs, tokenizer, nthreads=nthreads)

    all_preds = bert_cos_score_idf(
        model,
        refs,
        cands,
        tokenizer,
        idf_dict,
        verbose=False,
        device=device,
        batch_size=batch_size,
        all_layers=all_layers,
    ).cpu()

    if ref_group_boundaries is not None:
        max_preds = []
        for beg, end in ref_group_boundaries:
            max_preds.append(all_preds[beg:end].max(dim=0)[0])
        all_preds = torch.stack(max_preds, dim=0)


    #out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]  # P, R, F
    similarity = all_preds[..., 2].numpy()
    if use_bleu:
        cands_, refs_ = [list(ngrams(i, 1)) for i in cands], [list(ngrams(i, 1)) for i in refs] 
        diversity = [-sentence_bleu([c], r) for c, r in zip(cands_, refs_)]
    else:
        diversity = diverse(cands, refs)
    out = [x + diversity_factor*y for x, y in zip(similarity, diversity)]
    return out


