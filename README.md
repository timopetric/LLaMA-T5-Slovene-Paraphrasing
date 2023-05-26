# Natural language processing course 2022/23: `Paraphrasing sentences`

Team members:

- `Matej Kranjec`, `63200340`, `mk7972@student.uni-lj.si`
- `Timotej Petrič`, `63160264`, `tp1859@student.uni-lj.si`
- `Domen Vilar`, `63180310`, `dv6526@student.uni-lj.si`

Group public acronym/name: `skupina 8`

> This value will be used for publishing marks/scores. It will be known only to you and not you colleagues.

### Evaluation

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

### Report

Report is located in folder `report`
