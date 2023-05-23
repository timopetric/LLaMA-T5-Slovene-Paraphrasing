from score import score
import sys
sys.path.append("../paraphrasing")
from read_data import euparl

DATA_PATH = "../../data/euparl600k_ensl"
diversity_factor = 0.1

data = euparl(path=DATA_PATH, min_length=0, max_numbers=1e10, max_special_characters=1e10,filter_identical=False,shuffle=False)
n = len(data)

refs = data["original"][:n]
cands = data["translated"][:n]
scores = score(cands=cands, refs=refs, model_type="EMBEDDIA/sloberta", use_bleu=True)
file = open("parascores.out", "w")
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
