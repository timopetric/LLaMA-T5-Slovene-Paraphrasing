import os
from typing import Union, List, Tuple
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


def read(files: List[Tuple[str, str, str, Union[str, None]]] = [("../../data/euparl600k_ensl", "europarl-orig-sl-all.out", "europarl-tran-all.out", "parascores.out")],
         preprocess: Union[List[callable], None] = None,
         shuffle: bool = True,
         sort: bool = False,
         add_end_token=None,
         reverse_input_output=False,
         print_example_pair=False
        ) -> Dataset:
    
    original, translated, parascores = list(), list(), list()
    for path, orig_sl_filename, tran_sl_filename, parascore_filename in files:
        with open(os.path.join(path, orig_sl_filename)) as file:
            while True:
                l = file.readline()
                if not l: break
                original.append(l.strip("\n"))

        if parascore_filename is not None:
            with open(os.path.join(path, parascore_filename)) as file:
                while True:
                    l = file.readline()
                    if not l: break
                    parascores.append(float(l.strip("\n")))
        else:
            parascores += [1.0] * (len(original)-len(translated))

        with open(os.path.join(path, tran_sl_filename)) as file:
            while True:
                l = file.readline()
                if not l: break
                token_to_add = "" if add_end_token is None else add_end_token
                translated.append(l.strip("\n") + token_to_add)

    if preprocess:
        for p in preprocess:
            original, translated, parascores = p(original, translated, parascores)

    if reverse_input_output:
        original, translated = translated, original
        
    assert len(original) == len(translated) == len(parascores), "Lengths of original, translated and parascores must be equal."
        
    if print_example_pair:
        print("Example pair:")
        print("\tOriginal:  ", original[0])
        print("\tTranslated:", translated[0])
        print("\tParascore: ", parascores[0])
        print(f"Lenght of dataset: {len(original)}")
        print()

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



def euparl(min_length: int = 75,
           max_numbers: int = 5,
           max_special_characters: int = 5,
           max_length_diff: int = 25,
           min_parascore: float = 0.5,
           files: List[Tuple[str, str, str, Union[str, None]]] = [("../../data/euparl600k_ensl", "europarl-orig-sl-all.out", "europarl-tran-all.out", "parascores.out")],
           shuffle: bool = True,
           sort: bool = False,
           add_end_token=None,
           reverse_input_output=False,
           print_example_pair=False,
           filter=True
        ) -> Dataset:
    """
    Function reads data from given path, filters it and returns the result as a dataset.
    Parameters:
        -min_length: Minimum length of sentence in characters, default 75
        -max_numbers: Maximum amount of digits allowed in a sentence, default 5
        -max_special_characters: Maximum amount of allowed special characters in a sentence, default 5
        -max_length_diff: Maximum tolerated difference in length between a pair of sentences, default 25
        -min_parascore: Minimum parascore, default 0.5
        -files: List of tuples describing path to data, of form (path, orig_sl_filename, tran_sl_filename, parascore_filename or None)
        -shuffle: Shuffle the dataset, default True (Only one of sort and shuffle can be True)
        -sort: Sort the dataset by descending parascores, default False (Only one of sort and shuffle can be True)
        -add_end_token: Token to be added at the end of the translated sentences, default None 
        -reverse_input_output: Swaps the original and translated sentences, default False
        -print_example_pair: Prints a random pair of sentences, default False
        -filter: Whether to run any filtering at all, default True
    """
    
    # check that files in files exist
    for path, orig_sl_filename, tran_sl_filename, parascore_filename in files:
        assert os.path.exists(os.path.join(path, orig_sl_filename)), f"File {orig_sl_filename} does not exist in {path}."
        assert os.path.exists(os.path.join(path, tran_sl_filename)), f"File {tran_sl_filename} does not exist in {path}."
        if parascore_filename is not None:
            assert os.path.exists(os.path.join(path, parascore_filename)), f"File {parascore_filename} does not exist in {path}."
            
    if len(files) > 1:
        assert shuffle == True, "Shuffle must be True if more than one dataset is given in the files parameter."
    
    if filter:
        preprocess = list()
        preprocess.append(lambda x, y, z: remove_sentences_by_parascore(x, y, z, min_=min_parascore))
        preprocess.append(lambda x, y, z: remove_sentences_with_too_many_numbers(x, y, z, max_numbers))
        preprocess.append(lambda x, y, z: remove_sentences_with_too_many_special_characters(x, y, z, max_special_characters=max_special_characters))
        preprocess.append(lambda x, y, z: remove_short_sentences_by_chars(x, y, z, min_length))
        preprocess.append(lambda x, y, z: remove_sentences_with_different_lengths(x, y, z, max_length_diff))
    else:
        preprocess = None
    return read(
        files=files,
        preprocess=preprocess,
        shuffle=shuffle,
        sort=sort,
        add_end_token=add_end_token,
        reverse_input_output=reverse_input_output,
        print_example_pair=print_example_pair
    )

def euparl_(min_length: int = 75,
           max_numbers: int = 5,
           max_special_characters: int = 5,
           max_length_diff: int = 25,
           min_parascore: float = 0.5,
           path: str = "../../data/euparl600k_ensl",
           orig_sl_filename: str = "europarl-orig-sl-all.out",
           tran_sl_filename: str = "europarl-tran-all.out",
           parascore_filename: Union[str, None] = "parascores.out",
           shuffle: bool = True,
           sort: bool = False,
           add_end_token=None,
           reverse_input_output=False,
           print_example_pair=False,
           filter=True
        ) -> Dataset:
    return euparl(min_length=min_length,max_numbers=max_numbers,max_special_characters=max_special_characters,
                  max_length_diff=max_length_diff,min_parascore=min_parascore,files=[(path, orig_sl_filename, tran_sl_filename, parascore_filename)],
                  shuffle=shuffle, sort=sort, add_end_token=add_end_token, reverse_input_output=reverse_input_output,
                  print_example_pair=print_example_pair, filter=filter)


if __name__ == "__main__":
    dataset_files_list = []

    # # llama dataset
    # dataset_files_list.append((
    #     "/d/hpc/home/tp1859/nlp/opus/europarl-llama",
    #     "europarl-orig-sl.out",
    #     "europarl-llamapara-sl.out",
    #     "parascores.out"
    # ))
    # euparl translated dataset
    dataset_files_list.append((
        "/d/hpc/home/tp1859/nlp/opus/euparl600k_ensl",
        "europarl-orig-sl-all.out",
        "europarl-tran-all.out",
        "parascores.out"
    ))
    # MaCoCu translated dataset
    # dataset_files_list.append((
    #     "/d/hpc/projects/FRI/tp1859/nlp_project8/opus2/MaCoCu-sl-en_v1.0_slx2",
    #     "MaCoCu-sl-en_v1.0_slx2-orig-sl.out",
    #     "MaCoCu-sl-en_v1.0_slx2-tran-sl.out",
    #     "parascores.out"
    # ))
    
    for filter in (True, False):
        data = euparl(
            filter=filter,
            files=dataset_files_list,
        )
        print("$$$$$$$$ FILTER = ", str(filter), ", len: " + str(len(data)))
        print("\tdatasets:" + ", ".join([d[0] for d in dataset_files_list]))
    
    

    # print(len(data))
    # print(data["original"][:5])
    
    # for i in range(20):
    #     print(data["original"][i], "\n\t", data["translated"][i], "\n")
    
    exit(0)
        
    with open("mocacu_1000_orig_sl.txt", "a") as f, open("mocacu_1000_tran_sl.txt", "a") as g:
        for orig, tran in zip(data["original"], data["translated"][:1000]):
            f.write(orig + "\n")
            g.write(tran + "\n")

    exit(0)
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
