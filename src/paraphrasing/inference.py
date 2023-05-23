import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from glob import glob
from config import (
    T5_TASK_PREFIX,
    T5_INPUT_LEN_MAX,
    T5_TGT_LEN_MAX,
    MODEL_CHECKPOINT_FIN_GLOB,
    OUT_MODEL_CHECKPOINTS_DIR
)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def load_model(checkpoint_location):
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_location).to(device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_location)
    return model, tokenizer


def paraphrase(input_text, model, tokenizer):
    input_text_prefixed = T5_TASK_PREFIX + input_text
    inputs = tokenizer(input_text_prefixed, return_tensors="pt", max_length=T5_INPUT_LEN_MAX, truncation=True).to(device)

    # print(inputs)
    # print(tokenizer.decode(inputs["input_ids"][0].to("cpu"), skip_special_tokens=True))

    generated_outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=T5_TGT_LEN_MAX,
        # num_beams=4,
        # early_stopping=True,

        # num_beams=10,
        # num_beam_groups=5,
        # diversity_penalty=0.05,

        num_return_sequences=5,
        do_sample=True,
        top_k=200,
        top_p=0.95,
        # top_k=120,
        # top_p=0.98,
        early_stopping=True,
    )
    
    # print(generated_outputs)
    # exit()

    outputs = []
    for generated_output in generated_outputs:
        output_text = tokenizer.decode(generated_output, skip_special_tokens=True)
        outputs.append(output_text)
        
    return outputs


def main():
    # input_text = "Zato bi vas rada pozvala, minister, da v evropskem letu boja proti revščini in socialni izključenosti v Svetu in morda celo na ravni Evropskega sveta določite obvezne cilje za zmanjšanje stopnje osipa v šolah."
    # input_text = "Škisovo tržnico, največjo prireditev na prostem za mlade v Sloveniji, bodo otvorili tradicionalni mimohod študentskih klubov, nastop mažoretk in pozdravni govori."
    # input_text = "Program umetne inteligence ChatGPT podjetja OpenAI je po mesecu dni blokade v Italiji spet dovoljen, je danes sporočilo podjetje."
    input_text = "Evropski policijski urad Europol je konec marca posvaril pred zlorabami tovrstnih besedilnih generatorjev."
    
    checkpoint_location_glob = os.path.join(OUT_MODEL_CHECKPOINTS_DIR, MODEL_CHECKPOINT_FIN_GLOB)
    for checkpoint_location in glob(checkpoint_location_glob):
        print("##############################################")
        print("Loading model from:", checkpoint_location)
        try:
            model, tokenizer = load_model(checkpoint_location)
            output_texts = paraphrase(input_text, model, tokenizer)
        except Exception as e:
            print("Error:", e)
            continue

        print("########## Input text: ##########")
        print(input_text)
        print("########## Output text: ##########")
        for output_text in output_texts:
            print(output_text)
        print()


if __name__ == "__main__":
    main()

    
#srun -n1 -N1 -c8 -G1 --mem=24G --preserve-env --pty --partition=gpu singularity exec --nv /d/hpc/projects/FRI/tp1859/nlp_project8/lma/containers/hf.sif python3 inference.py
