# LLaMA model framework for paraphrasing of english sentences 

Paraphrase input sentences using the the [LLaMA model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) and [Vicuna](https://github.com/lm-sys/FastChat).

`cli.py` reads input from file with 1 sentence per line and outputs a Vicuna paraphrased sentence to processed/\*.out.

SLURM array directive supported (`run_llama.sbatch`).

For prompt template used look at `config.py`.

To setup make sure you have the following directories available here (can be symlinks):
+ `model_hf_vicuna`- used for Vicuna model weights (Instructions on on how to get the weights on site: [Vicuna](https://github.com/lm-sys/FastChat)).
+ `data` - corpuses in subfolders. Usef for `--file-in` in `cli.py`. This is the file with 1 sentence per line. This sentences get paraphrased.
+ `logs` - a directory that will hold the logs
+ `processed` - will be used for paraphrased sentences out
