import os

# MODEL_CHECKPOINT_TO_FINETUNE = "cjvt/t5-sl-large"
# BATCH_SIZE = 8   # works for now
# # BATCH_SIZE = 16 is too large for 32GB GPU memory

MODEL_CHECKPOINT_TO_FINETUNE = "cjvt/t5-sl-small"
# MODEL_CHECKPOINT_TO_FINETUNE = "/d/hpc/projects/FRI/tp1859/nlp_project8/t5_model_checkpoints/finetune_t5-sl-small_v0.0.6-LLaMApara-MaCoCu-slx2_b4_lr3E-05_g16_j38765749/checkpoint-12080"

BATCH_SIZE = 4
LEARING_RATE = 3e-5
# ADD_END_TOKEN = " </s>"  # ? TODO: used for testing, maybe not needed
ADD_END_TOKEN = None
REVERSE_INPUT_OUTPUT = False
GRADIENT_ACCUMULATION_STEPS = 16

######################### DATASET ###########################
# # llama dataset
# DATASET_PATH = "/d/hpc/home/tp1859/nlp/opus/europarl-llama"
# DATASET_ORIG_SENTS_FILE = "europarl-orig-sl.out"
# DATASET_TRAN_SENTS_FILE = "europarl-llamapara-sl.out"
# VERSION_DATASET_ENDING = "LLaMApara"

# # euparl translated dataset
# DATASET_PATH = "/d/hpc/home/tp1859/nlp/opus/euparl600k_ensl"
# DATASET_ORIG_SENTS_FILE = "europarl-orig-sl-all.out"
# DATASET_TRAN_SENTS_FILE = "europarl-tran-all.out"
# VERSION_DATASET_ENDING = "Euparl600k_ensl"

# MaCoCu translated dataset
DATASET_PATH = "/d/hpc/projects/FRI/tp1859/nlp_project8/opus2/MaCoCu-sl-en_v1.0_slx2"
DATASET_ORIG_SENTS_FILE = "MaCoCu-sl-en_v1.0_slx2-orig-sl.out"
DATASET_TRAN_SENTS_FILE = "MaCoCu-sl-en_v1.0_slx2-tran-sl.out"
VERSION_DATASET_ENDING = "MaCoCu-slx2"


dataset_files_list = []
# DATASET_FILES_LIST contains tuples of (path, orig_sents_file, tran_sents_file, parascores_file)
version_dataset_ending_list = []

# llama dataset
dataset_files_list.append((
    "/d/hpc/home/tp1859/nlp/opus/europarl-llama",
    "europarl-orig-sl.out",
    "europarl-llamapara-sl.out",
    "parascores.out"
))
version_dataset_ending_list.append("LLaMApara")

# # euparl translated dataset
# dataset_files_list.append((
#     "/d/hpc/home/tp1859/nlp/opus/euparl600k_ensl",
#     "europarl-orig-sl-all.out",
#     "europarl-tran-all.out",
#     "parascores.out"
# ))
# version_dataset_ending_list.append("Euparl600k_ensl")

# MaCoCu translated dataset
dataset_files_list.append((
    "/d/hpc/projects/FRI/tp1859/nlp_project8/opus2/MaCoCu-sl-en_v1.0_slx2",
    "MaCoCu-sl-en_v1.0_slx2-orig-sl.out",
    "MaCoCu-sl-en_v1.0_slx2-tran-sl.out",
    "parascores.out"
))
version_dataset_ending_list.append("MaCoCu-slx2")

#############################################################

DATASET_FILES_LIST = dataset_files_list
VERSION_DATASET_ENDING_LIST = "-".join(version_dataset_ending_list)

VERSION_STR = "0.0.6-" + VERSION_DATASET_ENDING_LIST
VERSION_STR += "-rev" if REVERSE_INPUT_OUTPUT else ""

# version history:
# 0.0.1-LLaMApara - uncleaned euparl-llama dataset (no cleaning, no deduplication)
# 0.0.2-LLaMApara - cleaned and filtered euparl-llama dataset
# 0.0.3-LLaMApara - cleaned and filtered dataset with parascore filtering
# 0.0.4-{ds name} - add saving last checkpoint at the end of training
#           + cleaned and filtered dataset with parascore filtering (default params for it)
#           + ds name = "LLaMApara" or "Euparl600k_ensl"
# 0.0.5-{ds name} - imporved dataset cleaning and filtering based on parascore
#           + dataset should be even smaller now ?
#           + increased number of epochs to 4 (from 3)
# 0.0.6-{ds name} - train with both MaCoCu and LLaMApara datasets at the same time


PRINT_EXAMPLE_PAIR = True

T5_INPUT_LEN_MAX = 512
T5_TGT_LEN_MAX = 512
T5_TASK_PREFIX = "parafraziraj: "

OUT_MODEL_CHECKPOINTS_DIR = "/d/hpc/projects/FRI/tp1859/nlp_project8/t5_model_checkpoints"
# MODEL_CHECKPOINT_FIN_GLOB = "finetune*/checkpoint-*"
# MODEL_CHECKPOINT_FIN_GLOB = "finetune_t5-sl-small_*_g8_*/checkpoint-*"
MODEL_CHECKPOINT_FIN_GLOB = "*v0.0.4*/checkpoint*fin"


MODEL_CHECKPOINT_TO_FINETUNE = os.getenv("NLP8_ENV_MODEL_CHECKPOINT_TO_FINETUNE", MODEL_CHECKPOINT_TO_FINETUNE)
BATCH_SIZE = int(os.getenv("NLP8_ENV_BATCH_SIZE", BATCH_SIZE))
LEARING_RATE = float(os.getenv("NLP8_ENV_LEARING_RATE", LEARING_RATE))
ADD_END_TOKEN = os.getenv("NLP8_ENV_ADD_END_TOKEN", ADD_END_TOKEN)
REVERSE_INPUT_OUTPUT = bool(int(os.getenv("NLP8_ENV_REVERSE_INPUT_OUTPUT", REVERSE_INPUT_OUTPUT)))  # 0 or 1
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("NLP8_ENV_GRADIENT_ACCUMULATION_STEPS", GRADIENT_ACCUMULATION_STEPS))
T5_INPUT_LEN_MAX = int(os.getenv("NLP8_ENV_T5_INPUT_LEN_MAX", T5_INPUT_LEN_MAX))
T5_TGT_LEN_MAX = int(os.getenv("NLP8_ENV_T5_TGT_LEN_MAX", T5_TGT_LEN_MAX))
T5_TASK_PREFIX = os.getenv("NLP8_ENV_T5_TASK_PREFIX", T5_TASK_PREFIX)
