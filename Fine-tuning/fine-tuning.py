#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
import warnings
from functools import partial


import matplotlib.pyplot as plt
import pandas as pd
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
)

import tokenfilter  



# LoRA hyperparameters
lora_r = 4
lora_alpha = 1 * lora_r
lora_mlp = True
lora_head = False  # currently unused, kept for completeness


target_layers = range(19, 29)  # 19–28


batch_size = 64
gradient_accumulation_steps = 8  
num_epochs = 15
learning_rate = 1e-4
weight_decay = 0.01
warmup_ratio = 0.1

# Paths & environment configuration
os.environ["HF_TOKEN"] = "your_token"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

model_path = "/root/autodl-fs/Mistral-7B-Instruct-v0.1"
glitch_token_csv_path = "/root/autodl-fs/glitchtokens/Mistral-7B-Instruct-v0.1-glitch-tokens-my.csv"

fix_rate_save_dir = "/root/autodl-fs/results/mistral"
eval_save_dir = "/root/autodl-fs/results/eval/mistral"

# Evaluation configuration
FIXED_SEED = 42
eval_tasks = ["mmlu", "gsm8k"]
eval_batch_size = 8


warnings.filterwarnings("ignore", message="Failed to get model SHA for")


def initialize_model(model_path: str, device: str = "auto", quant_type: str = "bfloat16"):
    """
    Initialize the causal LM model with optional quantization.

    Parameters
    ----------
    model_path : str
        Path or identifier of the pretrained model.
    device : str
        "auto", "cuda", "cuda:0", "cpu", etc.
    quant_type : str
        'bfloat16', 'float16', or other (default dtype).

    Returns
    -------
    torch.nn.Module
        Loaded model with gradients disabled on all parameters.
    """
    if device == "auto":
        device_map = "auto"
    elif device.startswith("cuda"):
        device_map = {"": device}
    else:
        device_map = {"": device}

    if quant_type == "bfloat16":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
    elif quant_type == "float16":
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
        )

    model.requires_grad_(False)
    return model



model = initialize_model(model_path, device="auto", quant_type="float32")
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = model.device


for param in model.parameters():
    param.requires_grad = False


df_glitch = pd.read_csv(glitch_token_csv_path)
glitchtokens = df_glitch["index"].tolist()


text_all = []
text_output_all = []
for glitchtoken in glitchtokens:
    token_str = tokenizer.decode([glitchtoken])
    prompt = (
        f"Question: Can you repeat the string '{token_str}' and return back to me?\n"
        f"Answer: Here is the repeated string:\n"
    )
    text_all.append(prompt)

    answer = f"'{token_str}'"
    text_output_all.append(answer)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


class GlitchTokenDataset(Dataset):
    def __init__(self, text_inputs, text_outputs, tokenizer, device, max_length: int = 384):
        self.inputs = text_inputs
        self.outputs = text_outputs
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        instruction = self.tokenizer(self.inputs[idx], add_special_tokens=False)
        response = self.tokenizer(self.outputs[idx], add_special_tokens=False)

        input_ids = (
            instruction["input_ids"]
            + response["input_ids"]
            + [self.tokenizer.eos_token_id]
        )
        attention_mask = (
            instruction["attention_mask"]
            + response["attention_mask"]
            + [1]
        )

        labels = (
            [-100] * len(instruction["input_ids"])
            + response["input_ids"]
            + [self.tokenizer.eos_token_id]
        )

        # 截断到 max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]

        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.tensor(attention_mask, device=self.device)
        labels = torch.tensor(labels, device=self.device)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    batch_input_ids = [item["input_ids"] for item in batch]
    batch_attention_mask = [item["attention_mask"] for item in batch]
    batch_labels = [item["labels"] for item in batch]

    max_length = max(len(ids) for ids in batch_input_ids)

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for input_ids, attention_mask, labels in zip(
        batch_input_ids, batch_attention_mask, batch_labels
    ):
        pad_len = max_length - len(input_ids)

        padded_input_ids.append(
            torch.cat(
                [
                    input_ids,
                    torch.full(
                        (pad_len,),
                        tokenizer.pad_token_id,
                        device=input_ids.device,
                    ),
                ]
            )
        )
        padded_attention_mask.append(
            torch.cat(
                [
                    attention_mask,
                    torch.zeros(pad_len, device=attention_mask.device),
                ]
            )
        )
        padded_labels.append(
            torch.cat(
                [
                    labels,
                    torch.full(
                        (pad_len,),
                        -100,
                        device=labels.device,
                    ),
                ]
            )
        )

    padded_input_ids = torch.stack(padded_input_ids)
    padded_attention_mask = torch.stack(padded_attention_mask)
    padded_labels = torch.stack(padded_labels)

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels,
    }


dataset = GlitchTokenDataset(
    text_all,
    text_output_all,
    tokenizer,
    device=model.device,
)

train_dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, device, dtype):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())

        A_tensor = torch.randn(in_dim, rank, device=device, dtype=dtype) * std_dev
        B_tensor = torch.zeros(rank, out_dim, device=device, dtype=dtype)

        self.A = torch.nn.Parameter(A_tensor)
        self.B = torch.nn.Parameter(B_tensor)
        self.rank = rank
        self.alpha = alpha

    def forward(self, x):
        return self.alpha / self.rank * (x @ self.A @ self.B)


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha, device, dtype):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha, device, dtype
        )

    def forward(self, x, config_flag=None):
        base_out = self.linear(x)
        if config_flag is not None:
            return base_out + config_flag * self.lora(x)
        return base_out + self.lora(x)

class ModelWithConfig(torch.nn.Module):
    def __init__(self, model, glitchtokens, tokenizer):
        super().__init__()
        self.model = model
        self.glitchtokens = glitchtokens
        self.tokenizer = tokenizer
        self.device = model.device
        self.config = model.config
        self.tie_weights = lambda: self

    def to(self, *args, **kwargs):
        return self.model.to(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def create_config_flag(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size = token_ids.shape[0]
        config_flag = torch.zeros(batch_size, device=self.model.device, dtype=self.model.dtype)

        for b in range(batch_size):
            if any(token.item() in self.glitchtokens for token in token_ids[b]):
                config_flag[b] = 1

        return config_flag.view(-1, 1, 1)

    def _patch_lora_forwards(self, config_flag):
        original_forwards = {}

        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx in target_layers and lora_mlp:
                key_gate = f"gate_proj_{layer_idx}"
                key_up = f"up_proj_{layer_idx}"

                original_forwards[key_gate] = layer.mlp.gate_proj.forward
                original_forwards[key_up] = layer.mlp.up_proj.forward

                layer.mlp.gate_proj.forward = partial(
                    original_forwards[key_gate], config_flag=config_flag
                )
                layer.mlp.up_proj.forward = partial(
                    original_forwards[key_up], config_flag=config_flag
                )

        return original_forwards

    def _restore_lora_forwards(self, original_forwards):
        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx in target_layers and lora_mlp:
                key_gate = f"gate_proj_{layer_idx}"
                key_up = f"up_proj_{layer_idx}"

                layer.mlp.gate_proj.forward = original_forwards[key_gate]
                layer.mlp.up_proj.forward = original_forwards[key_up]

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        config_flag = self.create_config_flag(input_ids)

        original_forwards = self._patch_lora_forwards(config_flag)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        if not hasattr(outputs, "logits") and isinstance(outputs, torch.Tensor):
            outputs.logits = outputs

        self._restore_lora_forwards(original_forwards)
        return outputs

    def generate(self, input_ids, **kwargs):
        config_flag = self.create_config_flag(input_ids)
        original_forwards = self._patch_lora_forwards(config_flag)

        outputs = self.model.generate(
            input_ids=input_ids,
            **kwargs,
        )

        self._restore_lora_forwards(original_forwards)
        return outputs

assign_lora = partial(
    LinearWithLoRA,
    rank=lora_r,
    alpha=lora_alpha,
    device=model.device,
    dtype=model.dtype,
)

for layer_idx, layer in enumerate(model.model.layers):
    if layer_idx in target_layers and lora_mlp:
        layer.mlp.gate_proj = assign_lora(layer.mlp.gate_proj)
        layer.mlp.up_proj = assign_lora(layer.mlp.up_proj)

wrapped_model = ModelWithConfig(model, glitchtokens, tokenizer)


def get_trainable_params(model: torch.nn.Module):
    trainable_params = []
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            trainable_params.append(param)
    return trainable_params


trainable_params = get_trainable_params(wrapped_model)
print(f"可训练参数数量: {len(trainable_params)}")


optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
num_training_steps = num_epochs * num_update_steps_per_epoch
num_warmup_steps = int(warmup_ratio * num_training_steps)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

epoch_losses = []
step_losses = []
steps = []

wrapped_model.train()
progress_bar = tqdm(range(num_training_steps))
completed_steps = 0

for epoch in range(num_epochs):
    total_loss = 0.0

    for batch_idx, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = wrapped_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss / gradient_accumulation_steps
        total_loss += loss.item() * gradient_accumulation_steps

        loss.backward()

        if (
            (batch_idx + 1) % gradient_accumulation_steps == 0
            or batch_idx == len(train_dataloader) - 1
        ):
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            step_losses.append(loss.item() * gradient_accumulation_steps)
            steps.append(completed_steps)

            completed_steps += 1
            progress_bar.update(1)
            progress_bar.set_description(
                f"Epoch {epoch + 1} - Loss: {loss.item() * gradient_accumulation_steps:.4f}"
            )

            if completed_steps % 100 == 0:
                print(
                    f"Step {completed_steps}: Loss = {loss.item() * gradient_accumulation_steps:.4f}"
                )

    avg_loss = total_loss / len(train_dataloader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{num_epochs} 完成，平均损失: {avg_loss:.4f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(steps, step_losses)
plt.title("Loss per Step during Training")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), epoch_losses, "bo-")
plt.title("Average Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.xticks(range(1, num_epochs + 1))
plt.grid(True)

plt.tight_layout()
plt.show()

print("训练完成！")

# ============================================================
# Glitch Token Repetition Check
# ============================================================

def repetive_judge(tokenizer, token_id, model_path, device):
    token = tokenizer.decode([token_id])
    token_without_space = tokenizer.decode([token_id]).lstrip()
    string_to_repeat = token_without_space

    if "Yi" in model_path:
        content = (
            f"Can you repeat the character '{token}' and return back to me?\n"
            f"Answer: Here is the repeated string:\n"
        )
        messages = [{"role": "user", "content": content}]
        input_ids = tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)

        output_ids = wrapped_model.generate(
            input_ids, max_new_tokens=10, do_sample=False, temperature=None, top_p=None
        )[0]
        all_response = tokenizer.decode(output_ids, skip_special_tokens=True)
        k = all_response.find("assistant") + len("assistant")
        response = all_response[k:]
        if token_without_space in response:
            return True
        else:
            return False

    else:
        text1 = "Question: Can you repeat the string '"
        text2 = "' and return back to me?\nAnswer: Here is the repeated string:\n"

        tokens1 = torch.tensor(tokenizer.encode(text1))
        tokens2 = torch.tensor(tokenizer.encode(text2))

        if "Qw" in model_path:  # qwen will not add special token in the beginning
            tokens = torch.cat(
                (tokens1, torch.tensor([token_id]), tokens2),
                dim=0,
            ).to(device)
        else:
            tokens = torch.cat(
                (tokens1, torch.tensor([token_id]), tokens2[1:]),
                dim=0,
            ).to(device)

        prompt_text = (
            f"Question: Can you repeat the string '{token}' and return back to me?\n"
            f"Answer: Here is the repeated string:\n"
        )
        k = len(prompt_text)

        tokens = torch.unsqueeze(tokens, dim=0)
        response_tokens = wrapped_model.generate(
            tokens,
            max_new_tokens=10,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_ids,
        )[0]
        all_response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        response = all_response[k:]
        if string_to_repeat in response:
            return True
        else:
            return False


total_tokens = len(glitchtokens)
passed_tokens = 0
passed_token_ids = []

for token_id in glitchtokens:
    if repetive_judge(tokenizer, token_id, model_path, device):
        passed_tokens += 1
        passed_token_ids.append(token_id)

    idx = glitchtokens.index(token_id)
    if (idx + 1) % 100 == 0:
        print(
            f"已通过{passed_tokens}/{idx + 1}，{passed_tokens / (idx + 1) * 100:.2f}%"
        )
        print(f"已处理 {idx + 1}/{total_tokens} 个tokens")

if passed_tokens > 0:
    print("通过检测的tokens示例:")
    for i, token_id in enumerate(passed_token_ids[:5]):
        token_str = tokenizer.decode([token_id])
        print(f"  {i + 1}. Token ID: {token_id}, Token: '{token_str}'")

    if passed_tokens > 5:
        print(f"  ... 以及其他 {passed_tokens - 5} 个tokens")

print(f"测试模型为{model_path}")
print(f"训练轮数为{num_epochs}")
print(f"训练批次大小为{batch_size * gradient_accumulation_steps}")
print(f"keylayer为{list(target_layers)}")
print(f"lora_r为{lora_r},lora_缩放因子为{lora_alpha}")
print(f"总共检测了 {total_tokens} 个glitch tokens")
print(
    f"通过repetive_judge检测的tokens数量: {passed_tokens} "
    f"({passed_tokens / total_tokens * 100:.2f}%)"
)

# ============================================================
# Save Fix Rate Results
# ============================================================

os.makedirs(fix_rate_save_dir, exist_ok=True)

results_data = {
    "model_path": model_path,
    "num_epochs": num_epochs,
    "batch_size": batch_size * gradient_accumulation_steps,
    "learning_rate": learning_rate,
    "target_layers": str(list(target_layers)),
    "lora_r": lora_r,
    "lora_alpha": lora_alpha,
    "total_tokens": total_tokens,
    "passed_tokens": passed_tokens,
    "fix_rate": passed_tokens / total_tokens * 100,
}

df_fix = pd.DataFrame([results_data])
fix_filename = f"r_{lora_r}_alpha_{lora_alpha}_fix_rate_results.csv"
fix_file_path = os.path.join(fix_rate_save_dir, fix_filename)
df_fix.to_csv(fix_file_path, index=False)
print(f"修复率结果已保存到: {fix_file_path}")


print(f"{FIXED_SEED}")

results = evaluator.simple_evaluate(
    model=HFLM(pretrained=wrapped_model, tokenizer=wrapped_model.tokenizer),
    tasks=eval_tasks,
    verbosity="WARNING",
    batch_size=eval_batch_size,
    random_seed=FIXED_SEED,
    numpy_random_seed=FIXED_SEED,
    torch_random_seed=FIXED_SEED,
    fewshot_random_seed=FIXED_SEED,
)

print(results["results"])
resulte = results["results"]

os.makedirs(eval_save_dir, exist_ok=True)

metrics = []
values = []
for task, task_results in resulte.items():
    for metric, value in task_results.items():
        if isinstance(value, (int, float)):
            metrics.append(f"{task}_{metric}")
            values.append(value)

df_eval = pd.DataFrame(
    {
        "metric": metrics,
        "value": values,
    }
)

eval_csv_path = os.path.join(
    eval_save_dir, f"r_{lora_r}_alpha_{lora_alpha}_fix_rate_results.csv"
)
df_eval.to_csv(eval_csv_path, index=False)
print(f"Results saved to {eval_csv_path}")
