
## config.py

ima več parametrov:

### UČENJE MODELA

DATASET_ORIG_SENTS_FILE = datoteka z izvornimi slovenski stavki od eu tolmača (1 poved na vrstico).

DATASET_TRAN_SENTS_FILE = datoteka s ciljnimi slovenski parafrazami (1 poved na vrstico).

DATASET_PATH = pot kjer se nahajata datoteki DATASET_ORIG_SENTS_FILE in DATASET_TRAN_SENTS_FILE


### INFERENCE

OUT_MODEL_CHECKPOINTS_DIR = kje se nahajajo checkpointi naučenih modelov

MODEL_CHECKPOINT_FIN_GLOB = "glob" string, ki pove katere checkpointe naložit in z njimi delati inference


#### Potek učenja t5 modelov je na voljo na strani (na voljo za ogled le za našo skupino):

https://wandb.ai/nlp8/huggingface?workspace=user-tp1859
