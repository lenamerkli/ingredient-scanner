from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from general import relative_path

import os
import json


MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True
BASE_MODEL_NAME = 'unsloth/Qwen2-0.5B-Instruct-bnb-4bit'


def main():
    # data = []
    # for file in os.listdir(relative_path('data/ingredients/synthetic')):
    #     if file.endswith('.json'):
    #         with open(relative_path(f"data/ingredients/synthetic/{file}"), 'r', encoding='utf-8') as f:
    #             data.append(json.load(f))
    # with open(relative_path('tmp/ingredients/train.jsonl'), 'w', encoding='utf-8') as f:
    #     for d in data:
    #         f.write(json.dumps(d) + '\n')
    dataset = load_dataset(
        'json',
        data_files=relative_path('data/ingredients/synthetic/train.jsonl'),
        split='train'
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj', ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias='none',  # Supports any, but = 'none' is optimized
        use_gradient_checkpointing=False,  # True or 'unsloth' for very long context
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field='text',
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=16,
            max_steps=256,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            output_dir='llm_models',
            optim='adamw_8bit',
            seed=3407,
        ),
    )
    trainer.train()
    model.save_pretrained_gguf('./llm_models', tokenizer, quantization_method='q4_k_m')


if __name__ == '__main__':
    main()
