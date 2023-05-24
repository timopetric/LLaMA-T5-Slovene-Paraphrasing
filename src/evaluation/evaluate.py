from score import score
import sys
sys.path.append("../paraphrasing")
from read_data import euparl
from tqdm import tqdm
import transformers

transformers.logging.set_verbosity_error()
DATA_PATH = "/d/hpc/home/tp1859/nlp/opus/euparl600k_ensl"
BATCH_SIZE_CPU=2048*8
BATCH_SIZE_GPU=1024
diversity_factor = 0.1

data = euparl(path=DATA_PATH, min_length=0, max_numbers=1e10, max_special_characters=1e10,filter_identical=False,shuffle=False)
n = len(data)
refs = data["original"][:n]
cands = data["translated"][:n]

# clear file
open("parascores.out", "w").close()

file = open("parascores.out", "a")
iter_range = range(0, len(refs), BATCH_SIZE_CPU)

for batch_start in tqdm(iter_range, desc="CPU batches"):
    scores = score(
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
