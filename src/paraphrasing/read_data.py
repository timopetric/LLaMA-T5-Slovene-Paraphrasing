import os
from typing import Union, List
from datasets.arrow_dataset import Dataset
import pandas as pd
import random
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


def read(path: str = "../../data/euparl600k_ensl",
         preprocess: Union[List[callable], None] = None,
         orig_sl_filename="europarl-orig-sl-all.out",
         tran_sl_filename="europarl-tran-all.out",
         add_end_token=None,
         reverse_input_output=False,
         print_example_pair=False) -> Dataset:
    original, translated = list(), list()
    with open(os.path.join(path, orig_sl_filename)) as file:
        while True:
           l = file.readline()
           if not l: break
           original.append(l.strip("\n"))
    with open(os.path.join(path, tran_sl_filename)) as file:
        while True:
           l = file.readline()
           if not l: break
           token_to_add = "" if add_end_token is None else add_end_token
           translated.append(l.strip("\n") + token_to_add)
    if preprocess:
        for p in preprocess:
            original, translated = p(original, translated)
            
    if reverse_input_output:
        original, translated = translated, original
        
    if print_example_pair:
        print("Example pair:")
        import random
        lin = random.randint(0, len(original))
        print("\tOriginal:  ", original[0])
        print("\tTranslated:", translated[0])
        print()

    df = pd.DataFrame()
    df["original"] = original
    df["translated"] = translated
    df = df.sample(frac=1, random_state=42)
    return Dataset.from_pandas(df)


def euparl(min_length: int = 50,
           max_numbers: int = 5,
           max_special_characters: int = 5,
           path: str = "../../data/euparl600k_ensl",
           orig_sl_filename="europarl-orig-sl-all.out",
           tran_sl_filename="europarl-tran-all.out",
           filter_identical = True,
           print_example_pair=False,
           add_end_token=None,
           reverse_input_output=False) -> Dataset:

    funcs_to_apply = [
        lambda x, y: remove_sentences_with_too_many_numbers(x, y, max_numbers),
        lambda x, y: remove_sentences_with_too_many_special_characters(x, y, max_special_characters),
        lambda x, y: remove_short_sentences_by_chars(x, y, min_length)
    ]
    if filter_identical:
        funcs_to_apply.append(lambda x, y: remove_identical_sentences(x, y))
        
    return read(
        path,
        funcs_to_apply,
        orig_sl_filename=orig_sl_filename,
        tran_sl_filename=tran_sl_filename,
        print_example_pair=print_example_pair,
        add_end_token=add_end_token,
        reverse_input_output=reverse_input_output)


if __name__ == "__main__":
    data = euparl(min_length=50, max_numbers=5, max_special_characters=5)
    lengths = [len(d["original"]) for d in data]
    plt.hist(lengths, bins="auto", label=f"max={max(lengths)}, min={min(lengths)}, mean={sum(lengths)/len(lengths)}")
    plt.legend(loc="upper right")
    plt.title(f"n={len(data)}")
    plt.xlabel("Length")
    plt.show()
    for i in range(10): print(data["original"][i], "\n", data["translated"][i], "\n")
