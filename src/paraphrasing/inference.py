import torch
from read_data import euparl
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset


#srun -n1 -N1 -c8 -G1 --mem=24G --preserve-env --pty --partition=gpu singularity exec /d/hpc/projects/FRI/tp1859/nlp_project8/lma/containers/hf.sif python3 inference.py

prefix = ""
max_input_length = 512
max_target_length = 128

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Load the trained model
model_path = "t5-sl-large-finetuned/checkpoint-59023"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

# Set up tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set up input text
input_text = "Zato bi vas rada pozvala, minister, da v evropskem letu boja proti revščini in socialni izključenosti v Svetu in morda celo na ravni Evropskega sveta določite obvezne cilje za zmanjšanje stopnje osipa v šolah."

# Tokenize the input
inputs = tokenizer(prefix + input_text, return_tensors="pt", max_length=max_input_length, truncation=True).to(device)

# Generate the output with the trained model
generated_outputs = model.generate(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"],
                                    max_length=max_target_length,
                                    num_beams=4,
                                    early_stopping=True)

# Decode the generated output
output_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)

print(output_text)