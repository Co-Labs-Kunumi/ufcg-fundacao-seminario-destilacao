from peft import PeftModel, PeftConfig, LoraConfig
from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
import torch
import colorama
from colorama import Fore, Style
# Local imports
from thinking_budget_processor import ThinkingTokenBudgetProcessor


def save_full_model(base_model_ckpt, model_ckpt):
    """
    Merge lora adapters into base model and save full version
    """
    nf4_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_ckpt,
        quantization_config=nf4_config,
        device_map="auto"
    )

    # Load LoRA weights into base model
    model = PeftModel.from_pretrained(base_model, model_ckpt)
    model = model.merge_and_unload()
    model.save_pretrained("./full_model_1", safe_serialization=True)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_ckpt)
    tokenizer.save_pretrained("./full_model_1")


def test_chat(model_ckpt, max_tokens, thinking_tokens):
    colorama.init()

    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        model_ckpt,
        torch_dtype="auto",
        device_map="auto"
    )

    while True:
        # Prepare input
        user_input = input(Style.RESET_ALL + ">> ")
        if user_input == "exit":
            break
        messages = [
            {"role": "user", "content": user_input}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        # Tokenize inputs and generate answer
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        processor = ThinkingTokenBudgetProcessor(
            tokenizer,
            max_thinking_tokens=thinking_tokens,
        )
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            logits_processor=[processor],
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(
            output_ids[:index],
            skip_special_tokens=True
        ).strip("\n")
        content = tokenizer.decode(
            output_ids[index:],
            skip_special_tokens=True
        ).strip("\n")

        print(Fore.GREEN + "thinking content:", thinking_content)
        print(Fore.RED + "content:", content)
