import os
from typing import Union, List
from datasets.arrow_dataset import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

def remove_short_sentences_by_chars(original, translated, parascores, min_length=30):
    new1, new2, new3 = list(), list(), list()
    for s1, s2, s3 in zip(original, translated, parascores):
        if len(s1) >= min_length and len(s2) >= min_length:
            new1.append(s1)
            new2.append(s2)
            new3.append(s3)
    return new1, new2, new3

    
def remove_sentences_with_too_many_numbers(original, translated, parascores, max_numbers=20):
    new1, new2, new3 = list(), list(), list()
    for s1, s2, s3 in zip(original, translated, parascores):
        if sum([s1.count(str(n)) for n in range(10)]) <= max_numbers and sum([s2.count(str(n)) for n in range(10)]) <= max_numbers:
            new1.append(s1)
            new2.append(s2)
            new3.append(s3)
    return new1, new2, new3


def remove_sentences_with_too_many_special_characters(original, translated, parascores, special_characters = "@#$%^&*()_+=[]{}\\|<>/", max_special_characters=20):
    new1, new2, new3 = list(), list(), list()
    for s1, s2, s3 in zip(original, translated, parascores):
        if sum([s1.count(c) for c in special_characters]) <= max_special_characters and sum([s2.count(c) for c in special_characters]) <= max_special_characters:
            new1.append(s1)
            new2.append(s2)
            new3.append(s3)
    return new1, new2, new3


def remove_sentences_by_parascore(original, translated, parascores, min_=0.0, max_=1.0):
    new1, new2, new3 = list(), list(), list()
    for s1, s2, s3 in zip(original, translated, parascores):
        if min_ <= s3 <= max_ :
            new1.append(s1)
            new2.append(s2)
            new3.append(s3)
    return new1, new2, new3


def remove_sentences_with_different_lengths(original, translated, parascores, max_diff=10):
    new1, new2, new3 = list(), list(), list()
    for s1, s2, s3 in zip(original, translated, parascores):
        if abs(len(s1)-len(s2))<=max_diff:
            new1.append(s1)
            new2.append(s2)
            new3.append(s3)
    return new1, new2, new3


def read(path: str = "../../data/euparl600k_ensl", preprocess: Union[List[callable], None] = None, shuffle: bool = True, sort: bool = False) -> Dataset:
    original, translated, parascores = list(), list(), list()
    with open(os.path.join(path, "europarl-orig-sl-all.out")) as file:
        while True:
           l = file.readline()
           if not l: break
           original.append(l.strip("\n"))
    with open(os.path.join(path, "europarl-tran-all.out")) as file:
        while True:
           l = file.readline()
           if not l: break
           translated.append(l.strip("\n"))
    with open(os.path.join(path, "parascores.out")) as file:
        while True:
           l = file.readline()
           if not l: break
           parascores.append(float(l.strip("\n")))
    if preprocess:
        for p in preprocess:
            original, translated, parascores = p(original, translated, parascores)
    df = pd.DataFrame()
    df["original"] = original
    df["translated"] = translated
    df["parascores"] = parascores
    assert (shuffle and sort) == False
    if shuffle:
        df = df.sample(frac=1, random_state=42)
    if sort:
        df = df.sort_values(by="parascores", ascending=False)
    return Dataset.from_pandas(df)


def euparl(min_length: int = 75, max_numbers: int = 5, max_special_characters: int = 5, max_length_diff: int = 25, min_parascore: float = 0.5, path: str = "../../data/euparl600k_ensl", shuffle = False, sort = True) -> Dataset:
    """
    Function reads data from given path, filters it and returns the result as a dataset.
    Parameters:
        -min_length: Minimum length of sentence in characters, default 75
        -max_numbers: Maximum amount of digits allowed in a sentence, default 5
        -max_special_characters: Maximum amount of allowed special characters in a sentence, default 5
        -max_length_diff: Maximum tolerated difference in length between a pair of sentences, default 25
        -min_parascore: Minimum parascore, default 0.5
        -path: Path to dataset folder
        -shuffle: Shuffle the dataset, default False (Only one of sort and shuffle can be True)
        -sort: Sort the dataset by descending parascores, default True (Only one of sort and shuffle can be True)
    """

    preprocess = list()
    preprocess.append(lambda x, y, z: remove_sentences_by_parascore(x, y, z, min_=min_parascore))
    preprocess.append(lambda x, y, z: remove_sentences_with_too_many_numbers(x, y, z, max_numbers))
    preprocess.append(lambda x, y, z: remove_sentences_with_too_many_special_characters(x, y, z, max_special_characters=max_special_characters))
    preprocess.append(lambda x, y, z: remove_short_sentences_by_chars(x, y, z, min_length))
    preprocess.append(lambda x, y, z: remove_sentences_with_different_lengths(x, y, z, max_length_diff))
    return read(path, preprocess=preprocess, shuffle=shuffle, sort=sort)


if __name__ == "__main__":
    data = euparl()
    print(len(data))
    plt.figure()
    plt.hist(data["parascores"], bins="auto")
    plt.show()
    for i in range(10): print(data["parascores"][i], "\n", data["original"][i], "\n", data["translated"][i], "\n")
    print("##########################\n")
    for i in range(1,11): print(data["parascores"][-i], "\n", data["original"][-i], "\n", data["translated"][-i], "\n")

    exit(0)
    data = euparl(min_length=50, max_numbers=5, max_special_characters=5)
    cands_, refs_ = [list(ngrams(i, 1)) for i in data["translated"]], [list(ngrams(i, 1)) for i in data["original"]] 
    diversity = [sentence_bleu([c], r) for c, r in tqdm(zip(cands_, refs_), total=len(cands_))]
    plt.hist(diversity, bins="auto")
    plt.show()
    exit(0)
    lengths = [len(d["original"]) for d in data]
    plt.hist(lengths, bins="auto", label=f"max={max(lengths)}, min={min(lengths)}, mean={sum(lengths)/len(lengths)}")
    plt.legend(loc="upper right")
    plt.title(f"n={len(data)}")
    plt.xlabel("Length")
    plt.show()
    for i in range(10): print(data["original"][i], "\n", data["translated"][i], "\n")