# Third party imports
import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EvalPrediction
)
from liger_kernel.transformers import (
    AutoLigerKernelForCausalLM,
    apply_liger_kernel_to_qwen3
)
from peft import (
    PeftModel, PeftConfig, LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)


def fine_tune_qlora(model_ckpt, dataset, liger_kernels=None, last_checkpoint=None):
    """
    Fine-tune model using custom training configuration and kernels
    """
    device = "cpu"
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    print(f"Using device: {device}")

    # Load student model. Save lora adapters and config
    model = get_qlora_model(model_ckpt, liger_kernels=liger_kernels)
    model.save_pretrained("./" + model_ckpt.split("/")[-1] + "-qlora")

    # Training arguments
    output_dir = "distill-" + model_ckpt.split("/")[-1] + "-poscomp"

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,

        num_train_epochs=3,
        warmup_steps=0,
        optim="adamw_bnb_8bit",
        weight_decay=0.001,
        learning_rate=2e-4,                # Recommended in qlora paper for
                                           # models below 33B.
        # Data preloading
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,

        # Evaluation, saving and logging
        # Batch size of 16 and this dataset gives around 75 steps total
        eval_strategy="steps",
        eval_steps=40,                    # Evaluate every n steps
        save_strategy="steps",
        save_steps=40,                    # Save checkpoint every n steps
        save_total_limit=2,                # Keep last 2 checkpoints
        logging_strategy="steps",
        logging_steps=5,
        logging_dir="./logs",
        report_to="tensorboard",

        # Optimizations
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,   # Effective batch size of 16
        gradient_checkpointing=False,     # Reduces memory usage
                                          # Decreases speed by 20%
        torch_compile=True,
        torch_compile_backend="inductor",
        # torch_empty_cache_steps=4,          # Reduces memory usage.
                                              # Decreases speed by 10%
        group_by_length=True,
    )

    # Trainer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=last_checkpoint)


def get_qlora_model(model_ckpt, liger_kernels):
    """
    Apply 4-bit quantization and lora adapters to model
    """
    if liger_kernels == "qwen3":
        apply_liger_kernel_to_qwen3(
            rope=True,
            rms_norm=True,
            swiglu=True,
            cross_entropy=True,
            fused_linear_cross_entropy=False,
        )

    nf4_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_ckpt,
        quantization_config=nf4_config
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model(model_ckpt, liger_kernels, compile_model):
    """
    Load model from hub for training. Apply kernels and compile.
    """
    config = AutoConfig.from_pretrained(model_ckpt)

    # Load base model
    if liger_kernels == "qwen3":
        apply_liger_kernel_to_qwen3(
            rope=True,
            swiglu=True,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=False
        )

    model = AutoModelForCausalLM.from_pretrained(model_ckpt, config=config)

    # Compile for performance
    model = torch.compile(model) if compile_model else model

    return model


def load_model_lora(model_ckpt, liger_kernels, compile_model):
    """
    Load lora model from hub for training. Apply kernels and compile.
    """
    peft_config = PeftConfig.from_pretrained(model_ckpt)

    # Load base model
    if liger_kernels == "qwen3":
        # Fused linear cross entropy is not working with LoRA
        apply_liger_kernel_to_qwen3(
            rope=True,
            swiglu=True,
            cross_entropy=True,
            fused_linear_cross_entropy=False,
            rms_norm=False
        )

    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path
    )

    # Compile for performance
    model = torch.compile(model) if compile_model else model

    # Load LoRA adapter
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    return model
