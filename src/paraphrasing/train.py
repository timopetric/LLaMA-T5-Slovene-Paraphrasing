from read_data import euparl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset

model_checkpoint = "cjvt/t5-sl-large"
prefix = ""

tokenizer = AutoTokenizer.from_pretrained("cjvt/t5-sl-large")

model = AutoModelForSeq2SeqLM.from_pretrained("cjvt/t5-sl-large")

max_input_length = 512
max_target_length = 128

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["original"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["translated"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs




if __name__ == "__main__":
    data = euparl()
    tokenized_data = data.map(preprocess_function, batched=True)
    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        f"{model_name}-finetuned",
        evaluation_strategy = "epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True,
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
        tokenizer=tokenizer)
    trainer.train()
