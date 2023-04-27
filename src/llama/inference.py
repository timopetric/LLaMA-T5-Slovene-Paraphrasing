"""Inference for FastChat models."""
import abc
from typing import Optional
import os
import warnings
import re
from tqdm import tqdm
import torch
from read_sentences import get_sentences_list
from config import (
    OUT_FILE_ORIG_FMT_STR,
    OUT_FILE_PARA_FMT_STR,
    SAVE_TO_FILE,
    OUT_DIR,
    PROMPT_USER_TEMPLATE)

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaTokenizer,
        LlamaForCausalLM,
        AutoModel,
        AutoModelForSeq2SeqLM,
    )
except ImportError:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LLaMATokenizer,
        LLamaForCausalLM,
        AutoModel,
        AutoModelForSeq2SeqLM,
    )

from fastchat.conversation import (
    conv_templates,
    get_default_conv_template,
    compute_skip_echo_len,
    SeparatorStyle,
)

from fastchat.serve.compression import compress_module
from fastchat.serve.monkey_patch_non_inplace import (
    replace_llama_attn_with_non_inplace_operations,
)
from fastchat.serve.serve_chatglm import chatglm_generate_stream


def raise_warning_for_old_weights(model_path, model):
    if "vicuna" in model_path.lower():
        try:
            is_vicuna = isinstance(model, LlamaForCausalLM)
        except Exception:
            is_vicuna = isinstance(model, LLamaForCausalLM)
        if is_vicuna and model.model.vocab_size > 32000:
            warnings.warn(
                "\nYou are probably using the old Vicuna-v0 model, "
                "which will generate unexpected results with the "
                "current fschat.\nYou can try one of the following methods:\n"
                "1. Upgrade your weights to the new Vicuna-v1.1: https://github.com/lm-sys/FastChat#vicuna-weights.\n"
                "2. Use the old conversation template by `python3 -m fastchat.serve.cli --model-path /path/to/vicuna-v0 --conv-template conv_one_shot`\n"
                "3. Downgrade fschat to fschat==0.1.10 (Not recommonded).\n"
            )


def get_gpu_memory(max_gpus=None):
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def load_model(
    model_path, device, num_gpus, max_gpu_memory=None, load_8bit=False, debug=False
):
    if device == "cpu":
        kwargs = {}
    elif device == "cuda":
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs["device_map"] = "auto"
                if max_gpu_memory is None:
                    kwargs[
                        "device_map"
                    ] = "sequential"  # This is important for not the same VRAM sizes
                    available_gpu_memory = get_gpu_memory(num_gpus)
                    kwargs["max_memory"] = {
                        i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                        for i in range(num_gpus)
                    }
                else:
                    kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}
        print("init_kwargs", kwargs)
    elif device == "mps":
        kwargs = {"torch_dtype": torch.float16}
        # Avoid bugs in mps backend by not using in-place operations.
        replace_llama_attn_with_non_inplace_operations()
    else:
        raise ValueError(f"Invalid device: {device}")

    if "chatglm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, **kwargs
        ).cuda()
    elif "google/flan-t5" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
    elif "dolly" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        # 50277 means "### End"
        tokenizer.eos_token_id = 50277
    elif "pythia" in model_path or "stablelm" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        raise_warning_for_old_weights(model_path, model)

    if load_8bit:
        compress_module(model, device)

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.to(device)

    if debug:
        print(model)

    return model, tokenizer


@torch.inference_mode()
def generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_ids", [tokenizer.eos_token_id])

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]

    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                encoder_outputs = model.encoder(
                    input_ids=torch.as_tensor([input_ids], device=device)
                )
                out = model(
                    torch.as_tensor([input_ids], device=device),
                    decoder_input_ids=torch.as_tensor(
                        [[model.generation_config.decoder_start_token_id]],
                        device=device,
                    ),
                    encoder_outputs=encoder_outputs,
                    use_cache=True,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model(
                    input_ids=torch.as_tensor([input_ids], device=device),
                    use_cache=True,
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=torch.as_tensor([[token]], device=device),
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            if stop_str:
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
            yield output

        if stopped:
            break

    del past_key_values


# # some gpt4 adjusted code. Not tested. possibli usefull so that the model.generator parameters would be used
# # instead of the direct model() approach as in generate_stream
# @torch.inference_mode()
# def generate_outputs(model, tokenizer, params, device, context_len=2048):
#     prompt = params["prompt"]
#     l_prompt = len(prompt)
#     temperature = float(params.get("temperature", 1.0))
#     max_new_tokens = int(params.get("max_new_tokens", 256))
#     stop_str = params.get("stop", None)
#     stop_token_ids = params.get("stop_ids", [tokenizer.eos_token_id])

#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#     max_src_len = context_len - input_ids.shape[-1] - max_new_tokens - 8
#     input_ids = input_ids[:, -max_src_len:]

#     generated_ids = model.generate(
#         input_ids=input_ids,
#         max_length=input_ids.shape[-1] + max_new_tokens,
#         temperature=temperature,
#         do_sample=True,
#         num_return_sequences=1,
#         eos_token_id=tokenizer.eos_token_id,
#         pad_token_id=tokenizer.pad_token_id,
#         bad_words_ids=[stop_token_ids] if stop_token_ids is not None else None,
#     )
#     output_ids = generated_ids[:, input_ids.shape[-1] :].tolist()[0]
#     return tokenizer.decode(output_ids)


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream, skip_echo_len: int):
        """Stream output."""


def chat_loop(
    model_path: str,
    device: str,
    num_gpus: str,
    max_gpu_memory: str,
    load_8bit: bool,
    conv_template: Optional[str],
    temperature: float,
    max_new_tokens: int,
    chatio: ChatIO,
    debug: bool,
    prompt_file: str,  # new argument
    corpus_name: str,  # new argument
):
    # Model
    model, tokenizer = load_model(
        model_path, device, num_gpus, max_gpu_memory, load_8bit, debug
    )

    # Chat
    if conv_template:
        conv = conv_templates[conv_template].copy()
    else:
        conv = get_default_conv_template(model_path).copy()

    # read all prompts from file
    print("Reading prompts from file...")
    prompts_all, batch_inx = get_sentences_list(prompt_file)
    print(f"Done reading prompts from file. {len(prompts_all)} prompts loaded.")
    
    out_filename_orig = OUT_FILE_ORIG_FMT_STR.format(corpus_name=corpus_name, batch_inx=batch_inx)
    out_filename_para = OUT_FILE_PARA_FMT_STR.format(corpus_name=corpus_name, batch_inx=batch_inx)
    out_filename_orig = os.path.join(os.getcwd(), OUT_DIR, out_filename_orig)
    out_filename_para = os.path.join(os.getcwd(), OUT_DIR, out_filename_para)

    print("Setting output files to:")
    print(f"Original prompts:    {out_filename_orig}")
    print(f"Paraphrased prompts: {out_filename_para}")

    with open(out_filename_orig, "w") as f_orig, \
         open(out_filename_para, "w") as f_para:
        f_orig.write("")
        f_para.write("")

    for prompt_user in tqdm(prompts_all, desc="Paraphrasing prompts"):

        # genrate new vicuna compatible prompt
        conv.messages = []
        prompt = PROMPT_USER_TEMPLATE.format(prompt_user=prompt_user)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        conv_prompt = conv.get_prompt()
        # TODO: move to func ^

        skip_echo_len = len(conv_prompt)

        params = {
            "model": model_path,
            "prompt": conv_prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else None,
        }

        output_stream = generate_stream(model, tokenizer, params, device)
        outputs = chatio.stream_output(output_stream, skip_echo_len)
        outputs = outputs.strip("\"' \n")

        if debug:
            print("################################################################################")
            print("########## full prompt")
            print(conv_prompt)
            print("########## prompt:")
            print(prompt_user)
            print("########## outputs:")
            print(outputs)
            print()
            # TODO: move to func ^

        if SAVE_TO_FILE:
            with open(out_filename_orig, "a") as f_orig, \
                 open(out_filename_para, "a") as f_para:

                f_orig.write(prompt_user + "\n")
                
                # replace all whitespaces with single space
                outputs_clean = re.sub(r"\s+", " ", outputs)
                f_para.write(outputs_clean + "\n")
