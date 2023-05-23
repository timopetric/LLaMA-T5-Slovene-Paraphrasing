import os

# MODEL_CHECKPOINT_TO_FINETUNE = "cjvt/t5-sl-large"
# BATCH_SIZE = 8   # works for now
# # BATCH_SIZE = 16 is too large for 32GB GPU memory

MODEL_CHECKPOINT_TO_FINETUNE = "cjvt/t5-sl-small"
BATCH_SIZE = 4
LEARING_RATE = 3e-5
# ADD_END_TOKEN = " </s>"  # ? TODO: used for testing, maybe not needed
ADD_END_TOKEN = None
REVERSE_INPUT_OUTPUT = False
GRADIENT_ACCUMULATION_STEPS = 16

VERSION_STR = "0.0.2-LLaMApara"
# version history:
# 0.0.1-LLaMApara - uncleaned euparl-llama dataset (no cleaning, no deduplication)
# 0.0.2-LLaMApara - cleaned and filtered euparl-llama dataset


DATASET_PATH = None # add path to dataset here
# DATASET_PATH = "/d/hpc/home/tp1859/nlp/nlp-course-skupina-8/src/translation/processed"
DATASET_PATH = "/d/hpc/home/tp1859/nlp/opus/europarl-llama"
DATASET_ORIG_SENTS_FILE = "europarl-orig-sl.out"
DATASET_TRAN_SENTS_FILE = "europarl-llamapara-sl.out"
PRINT_EXAMPLE_PAIR = True

OUT_MODEL_CHECKPOINTS_DIR = "/d/hpc/projects/FRI/tp1859/nlp_project8/t5_model_checkpoints"

T5_INPUT_LEN_MAX = 512
T5_TGT_LEN_MAX = 512
T5_TASK_PREFIX = "parafraziraj: "

# MODEL_CHECKPOINT_FIN_GLOB = "finetune*/checkpoint-*"
# MODEL_CHECKPOINT_FIN_GLOB = "finetune_t5-sl-small_*_g8_*/checkpoint-*"
MODEL_CHECKPOINT_FIN_GLOB = "finetune*3808023*/checkpoint-*"


MODEL_CHECKPOINT_TO_FINETUNE = os.getenv("NLP8_ENV_MODEL_CHECKPOINT_TO_FINETUNE", MODEL_CHECKPOINT_TO_FINETUNE)
BATCH_SIZE = int(os.getenv("NLP8_ENV_BATCH_SIZE", BATCH_SIZE))
LEARING_RATE = float(os.getenv("NLP8_ENV_LEARING_RATE", LEARING_RATE))
ADD_END_TOKEN = os.getenv("NLP8_ENV_ADD_END_TOKEN", ADD_END_TOKEN)
REVERSE_INPUT_OUTPUT = bool(int(os.getenv("NLP8_ENV_REVERSE_INPUT_OUTPUT", REVERSE_INPUT_OUTPUT)))  # 0 or 1
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("NLP8_ENV_GRADIENT_ACCUMULATION_STEPS", GRADIENT_ACCUMULATION_STEPS))
T5_INPUT_LEN_MAX = int(os.getenv("NLP8_ENV_T5_INPUT_LEN_MAX", T5_INPUT_LEN_MAX))
T5_TGT_LEN_MAX = int(os.getenv("NLP8_ENV_T5_TGT_LEN_MAX", T5_TGT_LEN_MAX))
T5_TASK_PREFIX = os.getenv("NLP8_ENV_T5_TASK_PREFIX", T5_TASK_PREFIX)
