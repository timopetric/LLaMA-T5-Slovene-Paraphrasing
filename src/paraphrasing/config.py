# MODEL_CHECKPOINT_TO_FINETUNE = "cjvt/t5-sl-large"
# BATCH_SIZE = 8   # works for now
# # BATCH_SIZE = 16 is too large for 32GB GPU memory

MODEL_CHECKPOINT_TO_FINETUNE = "cjvt/t5-sl-small"
# BATCH_SIZE = 16    # works for now
BATCH_SIZE = 32    # works for now

DATASET_PATH = None # add path to dataset here
OUT_MODEL_CHECKPOINTS_DIR = "/d/hpc/projects/FRI/tp1859/nlp_project8/t5_model_checkpoints"

T5_INPUT_LEN_MAX = 512
T5_TGT_LEN_MAX = 512
T5_TASK_PREFIX = "paraphrase: "

MODEL_CHECKPOINT_FIN = ""
