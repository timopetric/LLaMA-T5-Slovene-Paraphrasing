import os
from tqdm import tqdm
from config import *

"""
Basic function: Reads sentences from files and returns a list of sentences.
Supports SLURM array splitting to split the data into batches.
"""

# CORPUS_PATH_EN_ORIG = "opus/europarl/europarl-v7.sl-en.en"
# CORPUS_PATH_SL_ORIG = "opus/europarl/europarl-v7.sl-en.sl"
# CORPUS_PATH_PROCESSED_EN_ORIG = "processed/europarl-orig-all.out"
# CORPUS_PATH_PROCESSED_SL_ORIG = "processed/europarl-tran-all.out"
# corpus_name = "europarl"

CORPUS_PATH_EN_ORIG = "opus/MaCoCu-sl-en_v1.0/en_orig.in"
CORPUS_PATH_SL_ORIG = "opus/MaCoCu-sl-en_v1.0/sl_orig.in"
CORPUS_PATH_EN_ORIG_MISSING = "opus/MaCoCu-sl-en_v1.0/en_orig_missing.in"
CORPUS_PATH_SL_ORIG_MISSING = "opus/MaCoCu-sl-en_v1.0/sl_orig_missing.in"
CORPUS_PATH_PROCESSED_EN_ORIG = "processed/MaCoCu-sl-en_v1.0-orig-en.out"
CORPUS_PATH_PROCESSED_SL_ORIG = "processed/MaCoCu-sl-en_v1.0-orig-sl.out"
corpus_name = "MaCoCu-sl-en_v1.0_slx2"

# if SLURM vars are set we want to split the data into batches
batch_inx = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
num_batches = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))


def get_sentences_orig(path):
    """
    Returns a list of sentences from Europarl dataset.
    """
    sentences_list = []
    with open(path, "r") as f:
        for l in f.readlines():
            sentences_list.append(l.strip("\n"))
    return sentences_list


def remove_short_sentences_by_chars(sentences_list, min_length=30):
    new_sentences_list = []
    for s in sentences_list:
        if len(s) >= min_length:
            new_sentences_list.append(s)
    return new_sentences_list    

    
def remove_sentences_with_too_many_numbers(sentences_list, max_numbers=20):
    new_sentences_list = []
    for s in sentences_list:
        if sum([s.count(str(n)) for n in range(10)]) <= max_numbers:
            new_sentences_list.append(s)
    return new_sentences_list


def remove_sentences_with_too_many_special_characters(sentences_list, max_special_characters=20):
    new_sentences_list = []
    for s in sentences_list:
        if sum([s.count(c) for c in "!@#$%^&*()_+-=[]{};':\"\\|,.<>/?"]) <= max_special_characters:
            new_sentences_list.append(s)
    return new_sentences_list


def get_average_sentence_length_by_words(sentences_list):
    return sum([len(s.split()) for s in sentences_list]) / len(sentences_list)


def analyze_sentences_europarl(sentences_list):
    print("Number of sentences: " + str(len(sentences_list)))
    print("First 10 sentences:")
    for i in range(10):
        print("\t" + sentences_list[i])
    print("Last 10 sentences:")
    for i in range(10):
        print("\t" + sentences_list[-i])

        
def get_sentences_list(min_length=30, max_numbers=20, max_special_characters=20, max_sentences=-1):
    """
    Returns a list of sentences from Europarl dataset.
    param min_length: minimum length of a sentence in characters
    param max_numbers: maximum number of numbers in a sentence
    param max_special_characters: maximum number of special characters in a sentence
    return: list of sentences, batch_inx
    """
    
    sentences_list_ = get_sentences_orig(CORPUS_PATH_EN_ORIG)
    
    sentences_list_ = remove_short_sentences_by_chars(sentences_list_, min_length)
    sentences_list_ = remove_sentences_with_too_many_numbers(sentences_list_, max_numbers)
    sentences_list_ = remove_sentences_with_too_many_special_characters(sentences_list_, max_special_characters)
    if max_sentences > 0:
        sentences_list_ = sentences_list_[:max_sentences]  # only for testing

    # split the data into batches
    assert batch_inx < num_batches, f"If there are {num_batches} batches, batch_inx should be less than {num_batches}."
    s = len(sentences_list_) // num_batches
    return sentences_list_[batch_inx*s:(batch_inx+1)*s], batch_inx, corpus_name


def produce_sentence_pairs():
    with open(CORPUS_PATH_EN_ORIG, 'r') as f1, \
            open(CORPUS_PATH_SL_ORIG, 'r') as f2, \
            open(CORPUS_PATH_PROCESSED_EN_ORIG, 'r') as f3, \
            open(CORPUS_PATH_PROCESSED_SL_ORIG, 'w') as f4:

        sents_orig_en = f1.readlines()
        sents_orig_sl = f2.readlines()
        assert len(sents_orig_en) == len(sents_orig_sl)

        original_sentences = dict()
        for i in tqdm(range(len(sents_orig_en)), desc="Creating dictionary of original sentences"):
            original_sentences[sents_orig_en[i]] = i

        for line in tqdm(f3.readlines()):
            assert line in original_sentences, f"Line {line} not found in original sentences"
            inx = original_sentences[line]
            f4.write(sents_orig_sl[inx])


def get_missing_unprocessed_sentences():
    with open(CORPUS_PATH_EN_ORIG, 'r') as f_orig_en, \
            open(CORPUS_PATH_PROCESSED_EN_ORIG, 'r') as f_proc_en, \
            open(CORPUS_PATH_EN_ORIG_MISSING, 'w') as f_missing_en:
            
        sents_orig_processed_en = f_proc_en.readlines()
        processed_sentences = dict()
        for i in tqdm(range(len(sents_orig_processed_en)), desc="Creating dictionary of processed sentences"):
            processed_sentences[sents_orig_processed_en[i]] = i
            
        for line in tqdm(f_orig_en.readlines(), desc="Writing not translated sentences"):
            if line not in processed_sentences:
                f_missing_en.write(line)


def main():

    # get_sentences_europarl_orig_pairs()
    # get_missing_unprocessed_sentences()

    pass


if __name__ == "__main__":
    main()
    