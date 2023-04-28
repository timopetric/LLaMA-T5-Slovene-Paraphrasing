import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import (
    T5_TASK_PREFIX,
    T5_INPUT_LEN_MAX,
    T5_TGT_LEN_MAX,
    MODEL_CHECKPOINT_FIN
)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT_FIN).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT_FIN)


def paraphrase(input_text):
    input_text_prefixed = T5_TASK_PREFIX + input_text
    inputs = tokenizer(input_text_prefixed, return_tensors="pt", max_length=T5_INPUT_LEN_MAX, truncation=True).to(device)

    generated_outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=T5_TGT_LEN_MAX,
        num_beams=4,
        early_stopping=True
    )

    output_text = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
    return output_text


def main():
    print("Loading model from:", MODEL_CHECKPOINT_FIN)
    
    input_text = "Zato bi vas rada pozvala, minister, da v evropskem letu boja proti revščini in socialni izključenosti v Svetu in morda celo na ravni Evropskega sveta določite obvezne cilje za zmanjšanje stopnje osipa v šolah."

    output_text = paraphrase(input_text)

    print("########## Input text: ##########")
    print(input_text)
    print("########## Output text: ##########")
    print(output_text)


if __name__ == "__main__":
    main()
    
    
#srun -n1 -N1 -c8 -G1 --mem=24G --preserve-env --pty --partition=gpu singularity exec --nv /d/hpc/projects/FRI/tp1859/nlp_project8/lma/containers/hf.sif python3 inference.py
