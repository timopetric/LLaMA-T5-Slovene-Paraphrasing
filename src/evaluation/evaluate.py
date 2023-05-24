from score import score as sscore
import os
import sys
sys.path.append("../paraphrasing")
from read_data import euparl
from tqdm import tqdm
import transformers
import argparse

transformers.logging.set_verbosity_error()


def main(DATASET_PATH, DATASET_ORIG_SENTS_FILE, DATASET_TRAN_SENTS_FILE):
    BATCH_SIZE_CPU=2048*8
    BATCH_SIZE_GPU=1024
    diversity_factor = 0.1
    PARASCORES_OUT = os.path.join(DATASET_PATH, "parascores.out")

    print(f"Outputing scores to file: '{PARASCORES_OUT}'")

    data = euparl(
        path=DATASET_PATH,
        orig_sl_filename=DATASET_ORIG_SENTS_FILE,
        tran_sl_filename=DATASET_TRAN_SENTS_FILE,
        parascore_filename=None,
        min_length=0,
        max_numbers=1e10,
        max_special_characters=1e10,
        shuffle=False,
        filter=False
    )
    n = len(data)
    refs = data["original"][:n]
    cands = data["translated"][:n]

    # clear file
    open(PARASCORES_OUT, "w").close()

    file = open(PARASCORES_OUT, "a")
    iter_range = range(0, len(refs), BATCH_SIZE_CPU)

    for batch_start in tqdm(iter_range, desc="CPU batches"):
        scores = sscore(
            cands=cands[batch_start:batch_start+BATCH_SIZE_CPU],
            refs=refs[batch_start:batch_start+BATCH_SIZE_CPU],
            model_type="EMBEDDIA/sloberta",
            use_bleu=True,
            batch_size=BATCH_SIZE_GPU,
        )
        for s in scores:
            file.write(f"{s}\n")


    file.close()
    exit(0)

    results = list(zip(scores, cands, refs))
    results.sort(reverse=True)
    for score, orig, translated in results[:10]:
        print(score)
        print(orig)
        print(translated)
        print()
    print("#############################################\n")
    for score, orig, translated in results[-10:]:
        print(score)
        print(orig)
        print(translated)
        print()

if __name__ == "__main__":
    # DATASET_PATH = "/d/hpc/home/tp1859/nlp/opus/euparl600k_ensl"
    # DATASET_ORIG_SENTS_FILE = "europarl-orig-sl-all.out"
    # DATASET_TRAN_SENTS_FILE = "europarl-tran-all.out"

    # DATASET_PATH = "/d/hpc/home/tp1859/nlp/opus/europarl-llama"
    # DATASET_ORIG_SENTS_FILE = "europarl-orig-sl.out"
    # DATASET_TRAN_SENTS_FILE = "europarl-llamapara-sl.out"
    
    # read args and set vars with defaults
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset-path", required=False, default="/d/hpc/home/tp1859/nlp/opus/euparl600k_ensl", help="Path to dataset")
    ap.add_argument("-o", "--orig-sents-file", required=False, default="europarl-orig-sl-all.out", help="Original sentences file")
    ap.add_argument("-t", "--tran-sents-file", required=False, default="europarl-tran-all.out", help="Translated sentences file")
    
    args = vars(ap.parse_args())
    DATASET_PATH = args["dataset_path"]
    DATASET_ORIG_SENTS_FILE = args["orig_sents_file"]
    DATASET_TRAN_SENTS_FILE = args["tran_sents_file"]
    
    main(DATASET_PATH, DATASET_ORIG_SENTS_FILE, DATASET_TRAN_SENTS_FILE)
