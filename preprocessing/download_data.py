from transformers import Wav2Vec2Model
from datasets import list_datasets, load_dataset, list_metrics, load_metric
import numpy as np
import torch
import torchaudio
import requests
import matplotlib
import matplotlib.pyplot as plt
import IPython
import pickle as pkl
import datasets
import os
from tqdm import tqdm
from transformers import Wav2Vec2Model

counter = {'zh-HK': 5658, 'fy-NL': 5842, 'uk': 5915, 'fa': 6973, 'et': 6979, 'cs': 7558, 'pt': 7652, 'pl': 9271, 'tt': 9692, 'cy': 9899, 'ar': 10000, 'ca': 10000, 'de': 10000, 'en': 10000, 'eo': 10000, 'es': 10000, 'eu': 10000, 'fr': 10000, 'it': 10000, 'kab': 10000, 'nl': 10000, 'ru': 10000, 'rw': 10000, 'zh-CN': 10000}
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()

def open_dir(path):
  dir_of_res_path = f'/../data/sub_pickles/{path}'
  if not os.path.isdir(dir_of_res_path):
      os.makedirs(dir_of_res_path)

def save(data,k):
  path = f'/../data/sub_pickles/train/{lang}_{k}.pkl'
  with open(path,'wb') as f:
    pkl.dump(data, f)


def save_features_to_pkl(lang):
    open_dir(lang)
    window = 48000
    step = 32000
    dataset = load_dataset("common_voice", lang, split="train", streaming=True)
    dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
    dataset_iter = iter(dataset)
    data = []
    k = 1
    for sample in dataset_iter:
        array = sample["audio"]["array"]
        array = array[44:]
        pointer = 0
        while pointer + window < len(array):
            curr = array[pointer:pointer + window]
            tensor_data = torch.tensor([curr])
            with torch.inference_mode():
                features, _ = model(tensor_data)
                data.append(features)
            if k % 100 == 0 and k > 0:
                save(data, k)
                # print(f'save - {k}')
                data = []
            if (k == 5000):
                return
            k += 1

            pointer += step
    if len(data) > 0:
      save(data,k)
    print(f'finish - {lang} = {k}')


for lang in tqdm(counter.keys()):
    save_features_to_pkl(lang)