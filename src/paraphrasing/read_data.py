import os
from typing import Union, List
from datasets.arrow_dataset import Dataset
import pandas as pd
import matplotlib.pyplot as plt

def remove_short_sentences_by_chars(original, translated, min_length=30):
    new1, new2 = list(), list()
    for s1, s2 in zip(original, translated):
        if len(s1) >= min_length and len(s2) >= min_length:
            new1.append(s1)
            new2.append(s2)
    return new1, new2    

    
def remove_sentences_with_too_many_numbers(original, translated, max_numbers=20):
    new1, new2 = list(), list()
    for s1, s2 in zip(original, translated):
        if sum([s1.count(str(n)) for n in range(10)]) <= max_numbers and sum([s2.count(str(n)) for n in range(10)]) <= max_numbers:
            new1.append(s1)
            new2.append(s2)
    return new1, new2


def remove_sentences_with_too_many_special_characters(original, translated, special_characters = "@#$%^&*()_+=[]{}\\|<>/", max_special_characters=20):
    new1, new2 = list(), list()
    for s1, s2 in zip(original, translated):
        if sum([s1.count(c) for c in special_characters]) <= max_special_characters and sum([s2.count(c) for c in "!@#$%^&*()_+-=[]{};':\"\\|,.<>/?"]) <= max_special_characters:
            new1.append(s1)
            new2.append(s2)
    return new1, new2


def remove_identical_sentences(original, translated):
    new1, new2 = list(), list()
    for s1, s2 in zip(original, translated):
        if s1.lower() != s2.lower():
            new1.append(s1)
            new2.append(s2)
    return new1, new2

def read(path: str = "../../data/euparl600k_ensl", preprocess: Union[List[callable], None] = None) -> Dataset:
    original, translated = list(), list()
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
    if preprocess:
        for p in preprocess:
            original, translated = p(original, translated)
    df = pd.DataFrame()
    df["original"] = original
    df["translated"] = translated
    df = df.sample(frac=1, random_state=42)
    return Dataset.from_pandas(df)


def euparl(min_length: int = 50, max_numbers: int = 5, max_special_characters: int = 5, filter_identical = True, path: str = "../../data/euparl600k_ensl") -> Dataset:
    if filter_identical:
        return read(path, [lambda x, y: remove_sentences_with_too_many_numbers(x, y, max_numbers), lambda x, y: remove_sentences_with_too_many_special_characters(x, y, max_special_characters=max_special_characters), lambda x, y: remove_short_sentences_by_chars(x, y, min_length), lambda x, y: remove_identical_sentences(x, y)])
    return read(path, [lambda x, y: remove_sentences_with_too_many_numbers(x, y, max_numbers), lambda x, y: remove_sentences_with_too_many_special_characters(x, y, max_special_characters=max_special_characters), lambda x, y: remove_short_sentences_by_chars(x, y, min_length)])

if __name__ == "__main__":
    data = euparl(min_length=50, max_numbers=5, max_special_characters=5)
    lengths = [len(d["original"]) for d in data]
    plt.hist(lengths, bins="auto", label=f"max={max(lengths)}, min={min(lengths)}, mean={sum(lengths)/len(lengths)}")
    plt.legend(loc="upper right")
    plt.title(f"n={len(data)}")
    plt.xlabel("Length")
    plt.show()
    for i in range(10): print(data["original"][i], "\n", data["translated"][i], "\n")