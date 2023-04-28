import torch
from read_data import euparl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
import os
from config import (
    MODEL_CHECKPOINT_TO_FINETUNE,
    T5_TASK_PREFIX,
    BATCH_SIZE,
    T5_INPUT_LEN_MAX,
    T5_TGT_LEN_MAX,
    DATASET_PATH,
    OUT_MODEL_CHECKPOINTS_DIR,
)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_TO_FINETUNE)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT_TO_FINETUNE).to(device)


model_name = MODEL_CHECKPOINT_TO_FINETUNE.split("/")[-1]
out_checkpoints_dir = os.path.join(OUT_MODEL_CHECKPOINTS_DIR, f"finetuned_{model_name}_b{BATCH_SIZE}")
print("Using dataset from:", DATASET_PATH)
print("Setting output dir to:", out_checkpoints_dir)
print(f"Model '{model_name}' loaded:")
print(torch.cuda.is_available(),torch.cuda.device_count(),torch.cuda.current_device())
print(f"per_device_train_batch_size: {BATCH_SIZE}, per_device_eval_batch_size: {BATCH_SIZE}")
# print(model)


def preprocess_function(examples):
    inputs = [T5_TASK_PREFIX + doc for doc in examples["original"]]
    model_inputs = tokenizer(inputs, max_length=T5_INPUT_LEN_MAX, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["translated"], max_length=T5_TGT_LEN_MAX, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main():
    data = euparl(path=DATASET_PATH)
    tokenized_data = data.map(preprocess_function, batched=True)
    args = Seq2SeqTrainingArguments(
        out_checkpoints_dir,
        evaluation_strategy = "epoch",
        save_strategy="epoch",
        # save_steps=1000,  # if save_strategy="steps"
        save_total_limit=4,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        num_train_epochs=4,
        predict_with_generate=True,
    )
    
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
