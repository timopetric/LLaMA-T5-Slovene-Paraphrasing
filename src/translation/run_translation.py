# this is a modified file from 
# https://github.com/clarinsi/Slovene_NMT/commit/9641fb82371d538c0283641f18b29ddd30cd51f3#diff-791d4d41d3718d15d49180f3aacc8370b8cab07383f0d35b2713651cc0adfe46
# from the clarinsi / Slovene_NMT RSDO project

from platform import platform
from tqdm import tqdm, trange
# silence all tqdm progress bars
# from functools import partialmethod
# tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# from version import __version__
import arrow
from pydantic import BaseModel
from typing import Dict
from time import time
from glob import glob
from re import findall
import yaml
import os

import torch
from nemo.core.classes.modelPT import ModelPT
from nemo.utils import logging
import contextlib


if torch.cuda.is_available() and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
    logging.info("AMP enabled!\n")
    autocast = torch.cuda.amp.autocast
else:
    @contextlib.contextmanager
    def autocast():
        yield

from nltk import download, sent_tokenize
download('punkt')

_TEXT_LEN_LIMIT = 5000
_TEXT_SPLIT_THRESHOLD = 1024
_SPLIT_LEN = 512
_use_gpu_if_available = True

class NMTModel(BaseModel):
  class Config:
    arbitrary_types_allowed = True
  tag: str
  nemo: ModelPT
  platform: str
  active: int


start_time: str = None
models: Dict[str, Dict[str, NMTModel]] = {}
num_requests_processed: int = None


def translate_text(item):
  time0 = time()
  if item.src_language.lower() not in models:
    raise ValueError(f"Source language {item.src_language} unsupported")
  if item.tgt_language.lower() not in models[item.src_language.lower()]:
    raise ValueError(f"Target language {item.tgt_language} unsupported")

  logging.info(f" Q: {item.text}")

  if isinstance(item.text, str):
    text = [item.text]
  else:
    text = item.text
  text_len = sum(len(_text) for _text in text)
  if text_len > _TEXT_LEN_LIMIT:
    logging.warning(f'{text}, text length exceded {text_len}c [max {_TEXT_LEN_LIMIT}c]')
    raise ValueError(f"Bad request.")

  text_batch = []
  text_batch_split = []
  for _text in text:
    if len(_text) > _TEXT_SPLIT_THRESHOLD:
      _split_start = len(text_batch)
      _sent = sent_tokenize(_text)
      i = 0
      while i < len(_sent):
        j = i+1
        while j < len(_sent) and len(' '.join(_sent[i:j])) < _SPLIT_LEN: j+=1
        if len(' '.join(_sent[i:j])) > _TEXT_SPLIT_THRESHOLD:
          _split=findall(rf'(.{{1,{_SPLIT_LEN}}})(?:\s|$)',' '.join(_sent[i:j]))
          text_batch.extend(_split)
        else:
          text_batch.append(' '.join(_sent[i:j]))
        i = j
      _split_end = len(text_batch)
      text_batch_split.append((_split_start,_split_end))
    else:
      text_batch.append(_text)

  logging.debug(f' B: {text_batch}, BS: {text_batch_split}')

  if _use_gpu_if_available and torch.cuda.is_available():
      models[item.src_language.lower()][item.tgt_language.lower()].nemo = models[item.src_language.lower()][item.tgt_language.lower()].nemo.cuda()

  models[item.src_language.lower()][item.tgt_language.lower()].active += 1
  translation_batch = models[item.src_language.lower()][item.tgt_language.lower()].nemo.translate(text_batch)
  logging.debug(f' BT: {translation_batch}')
  models[item.src_language.lower()][item.tgt_language.lower()].active -= 1

  translation = []
  _start = 0
  for _split_start,_split_end in text_batch_split:
    if _split_start != _start:
      translation.extend(translation_batch[_start:_split_start])
    translation.append(' '.join(translation_batch[_split_start:_split_end]))
    _start = _split_end
  if _start < len(translation_batch):
    translation.extend(translation_batch[_start:])

  result = ' '.join(translation) if isinstance(item.text, str) else translation

  duration_seconds = time()-time0
  logging.debug(f' R: {result}')
  logging.debug(f'text_length: {text_len}c, duration: {duration_seconds:.2f}s')
  global num_requests_processed
  num_requests_processed += 1

  if num_requests_processed == 0:
    if _use_gpu_if_available and torch.cuda.is_available():
      # Force onto CPU
      models[item.src_language.lower()][item.tgt_language.lower()].nemo = models[item.src_language.lower()][item.tgt_language.lower()].nemo.cpu()
      torch.cuda.empty_cache()

  return result, text_len, duration_seconds


def initialize():
  time0 = time()
  models: Dict[str, Dict[str, NMTModel]] = {}
  for _model_info_path in glob(f"./models/**/model.info",recursive=True):
    with open(_model_info_path) as f:
      _model_info = yaml.safe_load(f)

    lang_pair = _model_info.get('language_pair', None)
    if lang_pair:
      _model_tag = f"{_model_info['language_pair']}:{_model_info['domain']}:{_model_info['version']}"
      _model_platform = "gpu" if _use_gpu_if_available and torch.cuda.is_available() else "cpu"
      _model_path = f"{os.path.dirname(_model_info_path)}/{_model_info['info']['framework'].partition(':')[-1].replace(':','_')}.{_model_info['info']['framework'].partition(':')[0]}"

      model = ModelPT.restore_from(_model_path,map_location="cuda" if _model_platform == "gpu" else "cpu")
      model.freeze()
      model.eval()

      if lang_pair != f"{model.src_language.lower()}{model.tgt_language.lower()}":
        logging.warning(f"Invalid model.info; language_pair '{lang_pair}', {_model_info['info']['framework'].partition(':')[-1].replace(':','_')}.{_model_info['info']['framework'].partition(':')[0]} '{model.src_language.lower()}{model.tgt_language.lower()}', unloading")
        del model
        continue

      models[model.src_language.lower()] = {}
      models[model.src_language.lower()][model.tgt_language.lower()] = NMTModel(
        tag = _model_tag,
        nemo = model,
        platform = _model_platform,
        active = 0,
      )

  logging.info(f'Loaded models {[ (models[src_lang][tgt_lang].tag,models[src_lang][tgt_lang].platform) for src_lang in models for tgt_lang in models[src_lang] ]}')
  logging.info(f'Initialization finished in {round(time()-time0,2)}s')

  start_time = arrow.utcnow().isoformat()
  num_requests_processed = 0
  return start_time, models, num_requests_processed


def main():
  from collections import namedtuple
  from pprint import pprint
  Item = namedtuple("Item", ["src_language", "tgt_language", "text"])
  
  repeat_times = 100
  duration_all = 0
  text_len_all = 0
  for _ in trange(repeat_times, desc=f"Translating {repeat_times} texts."):
    item = Item(
      src_language="en",
      tgt_language="sl",
      text="Once more unto the breach, dear friends, once more. Test sentence."
    )

    translated, text_len, duration_secs = translate_text(item)
    text_len_all += text_len
    duration_all += duration_secs
  
  pprint(translated)
  print(f"{repeat_times} translations took {duration_all:.2f}s or {duration_all/repeat_times:.3f}s per translation.")
  print(f"{repeat_times} translations had {text_len_all} chars or {text_len_all/repeat_times} average chars.")
  


if __name__ == "__main__":
  logging.setLevel(logging.INFO)
  start_time, models, num_requests_processed = initialize()
  
  main()
