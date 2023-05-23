from score import score
import sys
sys.path.append("../paraphrasing")
from read_data import euparl


diversity_factor = 0.2

data = euparl()
n = 200 #len(data)

refs = data["original"][:n]
cands = data["translated"][:n]
scores = score(cands=cands, refs=refs, model_type="EMBEDDIA/sloberta", diversity_factor=diversity_factor, use_bleu=True)

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
