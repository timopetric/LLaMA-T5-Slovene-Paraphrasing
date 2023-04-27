import os


# if SLURM vars are set we want to split the data into multiple batches
batch_inx = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
num_batches = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))

def _get_sentences_orig(path):
    """
    Returns a list of sentences from the dataset at path.
    """
    sentences_list = []
    with open(path, "r") as f:
        for l in f.readlines():
            sentences_list.append(l.strip("\n"))
    return sentences_list


def get_sentences_list(file_in):
    sentences_list = _get_sentences_orig(file_in)
    
    # split the data into batches
    assert batch_inx < num_batches, f"If there are {num_batches} batches, batch_inx should be less than {num_batches}."
    s = len(sentences_list) // num_batches
    return sentences_list[batch_inx*s:(batch_inx+1)*s], batch_inx

