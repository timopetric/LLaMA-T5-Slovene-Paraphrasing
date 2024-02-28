# LLaMA-T5-Slovene-Paraphrasing

### Natural Language Processing course 2022-23 at Faculty of Computer and Information Science at University of Ljubljana

This is a NLP course project in which we explored ways of Slovene language sentence paraphrisation.
The best approach turned out to be traslating original Slovene sentence into English, use Vicuna/LLaMA (1st gen) to paraphrase the sentence and then translate the results back into Slovene.

### The [PDF report describing the methodology is located here](report/NLP_project_Paraphrasing_Sentences.pdf).

### Team members:

- `Matej Kranjec`
- `Timotej Petrič`
- `Domen Vilar`

#### 

## Evaluation

#### plot_distribution_scores.py

This script plots the distribution of scores and writes the best paraphrases to a file based on the provided scores and paraphrases directories.

Arguments:

- `--scores_path`: Path to the directory containing the scores files.
- `--paraphrases_path`: Path to the directory containing the paraphrases files.
- `--output_directory`: Output directory for the best paraphrases.

<!-- The script can be run on an HPC system using the following command:

~~~
srun -n1 -N1 -c8 -G1 --mem=32G --time=1-00:00 --preserve-env --pty --partition=gpu singularity exec --nv /d/hpc/projects/FRI/tp1859/nlp_project8/lma/containers/hf.sif python3 plot_distribution_scores.py --scores_path <scores_directory> --paraphrases_path <paraphrases_directory> --output_directory <output_directory>
~~~

-->

Output:

- The best paraphrases will be written to a file named "best_paraphrases_euparl_t5.txt" in the specified output directory.
- Two distribution plots will be generated: "maximum_values.png" and "first_values.png" in the current directory where script will run.

#### manual_evaluation.py

This script is used for manual evaluation of the best paraphrases.

Arguments:

The script accepts the following arguments:

- `--file1_path`: Path to the first file containing paraphrases.
- `--file2_path`: Path to the second file containing paraphrases.
- `--file_original_path`: Path to the file containing original sentences.
- `--scores_folder_path`: Path to the folder where the evaluation scores will be stored.

Note: If the specified `scores_folder_path` does not exist, the script will create it.

#### evaluation/run.sh

Each dataset block in the run.sh file contains the necessary variables for the evaluation. Make sure to modify these variables according to your dataset paths and filenames.

For example, if you want to evaluate the `europarl-llama` dataset, uncomment the block of code for that dataset and update the following variables:

- `DATASET_PATH` variable represents the path to the dataset you want to evaluate.
- `DATASET_ORIG_SENTS_FILE` variable represents the file containing the original sentences of the dataset.
- `DATASET_TRAN_SENTS_FILE` variable represents the file containing the paraphrased sentences of the dataset.

Once you have made the necessary modifications, you can run the run.sh script. It will execute the evaluate.py script with the provided dataset variables.

If you prefer to run the evaluation separately without using the run.sh script, you can directly execute the evaluate.py script and pass the dataset variables as command-line arguments.


## Run the T5 model

\*You have to have `transformers` python library installed. Preferrably the one with GPU/CUDA support.
\*The Singularity (Docker) image with the prepared env is already set up in the shared location `/d/hpc/projects/FRI/tp1859/nlp_project8/lma/containers/hf.sif` on _Arnes HPC_.

1. Move to `src/paraphrasing` directory.
1. Download the `finetune_t5-sl-small_v0.0.4-Euparl600k_ensl_b4_lr3E-05_g16_j38753698` model from: 
https://unilj-my.sharepoint.com/:f:/g/personal/tp1859_student_uni-lj_si/Eie-WJrrsIVAiJCFNQ8r28UBhKVq6vxhvNcud7RgXTr0tw?e=Xot15v
2. Rename the downloaded folder to `models/t5model`.
3. change `OUT_MODEL_CHECKPOINTS_DIR` in `config.py` to `models`.
4. change `MODEL_CHECKPOINT_FIN_GLOB` in `config.py` to `t5*`.
5. run `python3 inference.py` or `sbatch run.sbatch` if running on _Arnes HPC_.

## Run the LLaMA/Vicuna based model

\*You have to have `transformers` python library installed. Preferrably the one with GPU/CUDA support.
\*The Singularity (Docker) image with the prepared env is already set up in the shared location `/d/hpc/projects/FRI/tp1859/nlp_project8/lma/containers/hf.sif` on _Arnes HPC_.

Since the model weights setup is quite difficult to do and the weights transformations consume large amounts of disk (at least 60 GB with intermediate cleanups) and RAM (60GB) we have prepared the already converted weights and uploaded them to the shared _Arnes HPC_ space at path: `/d/hpc/projects/FRI/tp1859/nlp_project8/lma/model_hf_vicuna`. The Vicuna/LLaMA 13B model is in half precision and takes about 25 GB of VRAM on the GPU.

You can then run the paraphrase generation by modifying the `run_llama.sbatch` script in `src/llama` directory. You should set `--model-path` to the above LLaMA model wights directory.
Then change the `--corpus-name` to some string and set `--file-in` to a file containing 1 english sentence per line. Then you can run the inference by running `sbatch run_llama.sbatch`. Output files will be in `processed` directory, logs in `logs`.

### Translation

You can run translation with NEmO model simillarly as described in T5 / Vicuna section. Needed files like `run_translation.sbatch` are in `src/translation` dir.


### Report

Report is located in folder `report`
