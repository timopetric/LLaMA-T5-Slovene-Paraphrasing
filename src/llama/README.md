# LLaMA model framework for parafrization of english sentences 

One way to paraphrase input sentences using the the [LLaMA model](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/) and [Vicuna](https://github.com/lm-sys/FastChat).

To setup make sure you have the following directories available here (can be symlinks):
+ `model_hf_vicuna`- used for Vicuna model weights (Instructions on on how to get the weights on site: [Vicuna](https://github.com/lm-sys/FastChat)).
+ `data` - corpuses in subfolders. Usef for `--file-in` in `cli.py`. This is the file with 1 sentence per line. This sentences get paraphrased.
+ `logs` - a directory that will hold the logs
+ `processed` - will be used for paraphrased sentences out
