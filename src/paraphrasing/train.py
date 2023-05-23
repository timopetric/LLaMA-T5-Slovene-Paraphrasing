from config import (
    MODEL_CHECKPOINT_TO_FINETUNE,
    T5_TASK_PREFIX,
    BATCH_SIZE,
    T5_INPUT_LEN_MAX,
    T5_TGT_LEN_MAX,
    DATASET_PATH,
    DATASET_ORIG_SENTS_FILE,
    DATASET_TRAN_SENTS_FILE,
    PRINT_EXAMPLE_PAIR,
    OUT_MODEL_CHECKPOINTS_DIR,
    LEARING_RATE,
    ADD_END_TOKEN,
    REVERSE_INPUT_OUTPUT,
    GRADIENT_ACCUMULATION_STEPS,
    VERSION_STR,
)
import os
import torch
from read_data import euparl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_TO_FINETUNE)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT_TO_FINETUNE).to(device)


def preprocess_function(examples):
    inputs = [T5_TASK_PREFIX + doc for doc in examples["original"]]
    model_inputs = tokenizer(inputs, max_length=T5_INPUT_LEN_MAX, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["translated"], max_length=T5_TGT_LEN_MAX, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():    
    model_name = MODEL_CHECKPOINT_TO_FINETUNE.split("/")[-1]
    checkpoints_name = "finetune_{}_v{}_b{}_lr{:.0E}_g{}_j{}".format(
        model_name,
        VERSION_STR,
        BATCH_SIZE,
        LEARING_RATE,
        GRADIENT_ACCUMULATION_STEPS,
        os.getenv('SLURM_JOB_ID', '0'),
    )
    out_checkpoints_dir = os.path.join(OUT_MODEL_CHECKPOINTS_DIR, checkpoints_name)

    print(f"Model to load: '{model_name}'.")
    print(f"Using dataset from: '{DATASET_PATH}'.")
    print(f"Using prefix: '{T5_TASK_PREFIX}'.")
    print(f"add_end_token: '{ADD_END_TOKEN}'.")
    print(f"reverse_input_output: '{REVERSE_INPUT_OUTPUT}'.")
    print(f"gradient_accumulation_steps: '{GRADIENT_ACCUMULATION_STEPS}'.")
    print(f"learning_rate: {LEARING_RATE}")
    print(f"Using device: '{device}'.")
    print(f"Setting output dir to: '{out_checkpoints_dir}'.")
    print(f"per_device_train_batch_size: {BATCH_SIZE}, per_device_eval_batch_size: {BATCH_SIZE}")
    # print(model)
    
    print(DATASET_ORIG_SENTS_FILE, DATASET_TRAN_SENTS_FILE)
    
    data = euparl(
        path=DATASET_PATH,
        orig_sl_filename=DATASET_ORIG_SENTS_FILE,
        tran_sl_filename=DATASET_TRAN_SENTS_FILE,
        print_example_pair=PRINT_EXAMPLE_PAIR,
        add_end_token=ADD_END_TOKEN,
        reverse_input_output=REVERSE_INPUT_OUTPUT,
    )
    tokenized_data = data.map(preprocess_function, batched=True)
    
    args = Seq2SeqTrainingArguments(
        out_checkpoints_dir,
        evaluation_strategy = "epoch",
        save_strategy="epoch",
        num_train_epochs=3,
        load_best_model_at_end=True,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=1000,
        save_total_limit=4,
        learning_rate=LEARING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        # weight_decay=1e-4,
        predict_with_generate=True,
        run_name=checkpoints_name,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        eval_accumulation_steps=1,
        generation_max_length=T5_TGT_LEN_MAX,
    )

    print("Training arguments:")
    print(args)

    split = int(len(tokenized_data)*0.8)
    train_data, test_data = Dataset.from_dict(tokenized_data[0:split]), Dataset.from_dict(tokenized_data[split:])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
