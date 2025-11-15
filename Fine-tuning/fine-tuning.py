import torch
from torch.optim import AdamW
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
import os
from torch.utils.data import Dataset, DataLoader
import tokenfilter
import pandas as pd
import warnings
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


# default hyperparameter choices
lora_r = 4
lora_alpha = 1 * lora_r
lora_mlp = True
lora_head = False

# 定义要替换的层的范围
target_layers = range(19, 29)  # 这将替换第19层到第28层

batch_size = 64
# 设置梯度累积步数
gradient_accumulation_steps = 8  # 相当于将batch size增大4倍
# 训练参数设置
num_epochs = 15
learning_rate = 1e-4
weight_decay = 0.01
warmup_ratio = 0.1



def initialize_model_and_tokenizer(model_path, device="auto", quant_type="bfloat16"):
    # 设置设备映射
    if device == "auto":
        device_map = "auto"  # 使用所有可用的GPU
    elif device.startswith("cuda"):
        device_map = {"": device}  # 使用指定的单个GPU
    else:
        device_map = {"": device}  # 使用CPU或其他指定设备
    
    # 加载模型和分词器
    
    if quant_type == 'bfloat16':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
    elif quant_type == 'float16':
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

os.environ["HF_TOKEN"] = 'your_token'
model_path = "/root/autodl-fs/Mistral-7B-Instruct-v0.1"
model = initialize_model_and_tokenizer(model_path, device="auto", quant_type="float32")
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = model.device
for param in model.parameters():
    param.requires_grad = False

import pandas as pd
file_path = '/root/autodl-fs/检测的glitchtokens/Mistral-7B-Instruct-v0.1-glitch-tokens-my.csv'
df = pd.read_csv(file_path)
glitchtokens = df['index'].tolist()


# 假设已有text_all和text_output_all列表

text_all = []
text_output_all = []
for glitchtoken in glitchtokens:
    token = tokenizer.decode([glitchtoken])
    text = f"Question: Can you repeat the string '{token}' and return back to me?\nAnswer: Here is the repeated string:\n"
    text_all.append(text)

    text_output = f"'{token}'"
    text_output_all.append(text_output)


# 检查是否已有pad_token，如果没有才设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("设置pad_token为eos_token")
else:
    print(f"已存在pad_token: {tokenizer.pad_token}")

# 3. 创建自定义数据集
class GlitchTokenDataset(Dataset):
    def __init__(self, text_inputs, text_outputs, tokenizer, device, max_length=384):
        self.inputs = text_inputs
        self.outputs = text_outputs
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        # 构建指令格式
        instruction = self.tokenizer(self.inputs[idx], add_special_tokens=False)
        response = self.tokenizer(self.outputs[idx], add_special_tokens=False)
        
        # 构建输入 - 包含指令和回答
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.eos_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        
        # 构建标签 - 指令部分用-100标记，只计算回答部分的损失
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.eos_token_id]
        
        # 如果超过最大长度则截断
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
        
        # 转换为tensor并移至设备
        input_ids = torch.tensor(input_ids, device=self.device)
        attention_mask = torch.tensor(attention_mask, device=self.device)
        labels = torch.tensor(labels, device=self.device)
        

        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

# 4. 创建数据集实例
dataset = GlitchTokenDataset(
    text_all, 
    text_output_all, 
    tokenizer,
    device=model.device,  # 传入模型所在的device

)

def collate_fn(batch):
    """
    自定义的collate函数，用于处理不同长度的序列
    """
    # 将批次中的每个键分别收集
    batch_input_ids = [item['input_ids'] for item in batch]
    batch_attention_mask = [item['attention_mask'] for item in batch]
    batch_labels = [item['labels'] for item in batch]

    
    # 对input_ids、attention_mask和labels进行padding
    max_length = max(len(ids) for ids in batch_input_ids)
    
    # 创建padding后的张量
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    
    # 填充每个样本
    for input_ids, attention_mask, labels in zip(batch_input_ids, batch_attention_mask, batch_labels):
        # 计算需要填充的长度
        pad_len = max_length - len(input_ids)
        
        # 填充input_ids
        padded_input_ids.append(torch.cat([
            input_ids,
            torch.full((pad_len,), tokenizer.pad_token_id, device=input_ids.device)
        ]))
        
        # 填充attention_mask
        padded_attention_mask.append(torch.cat([
            attention_mask,
            torch.zeros(pad_len, device=attention_mask.device)
        ]))
        
        # 填充labels，使用-100填充，确保不计算损失
        padded_labels.append(torch.cat([
            labels,
            torch.full((pad_len,), -100, device=labels.device)
        ]))
    
    # 堆叠形成批次
    padded_input_ids = torch.stack(padded_input_ids)
    padded_attention_mask = torch.stack(padded_attention_mask)
    padded_labels = torch.stack(padded_labels)
    
    return {
        'input_ids': padded_input_ids,
        'attention_mask': padded_attention_mask,
        'labels': padded_labels,
    }

# 5. 创建DataLoader

train_dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn  # 使用自定义的collate_fn
)
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha,device,dtype):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        # 创建张量，将其移动到正确的设备，然后把它包装为Parameter
        A_tensor = torch.randn(in_dim, rank, device=device,dtype = dtype) * std_dev
        B_tensor = torch.zeros(rank, out_dim, device=device,dtype = dtype)
        
        # 将张量转换为Parameter，自动设置requires_grad=True
        self.A = torch.nn.Parameter(A_tensor)
        self.B = torch.nn.Parameter(B_tensor)
        self.rank = rank
        self.alpha = alpha
    def forward(self, x):
        x = self.alpha/self.rank * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha, device, dtype):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha, device, dtype
        )

    def forward(self, x, config_flag=None):
        # 使用传入的config_flag而不是全局变量
        return self.linear(x) + (config_flag * self.lora(x) if config_flag is not None else self.lora(x))

class ModelWithConfig(torch.nn.Module):
    def __init__(self, model, glitchtokens,tokenizer):
        super().__init__()
        self.model = model
        self.glitchtokens = glitchtokens
        self.tokenizer = tokenizer
        self.device = model.device
        self.config = model.config
        self.tie_weights = lambda: self



    # 确保train和eval方法正确传递
    def to(self, *args, **kwargs):
            return self.model.to(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self
        
    def create_config_flag(self, token_ids):
        # 创建批量大小的零张量，用于标记每个批次是否包含glitchtokens
        batch_size = token_ids.shape[0]
        config_flag = torch.zeros(batch_size, device=self.model.device,dtype=model.dtype)

        # 检查每个批次中是否有token在glitchtokens列表中
        for b in range(batch_size):
            # 如果该批次中任何一个token在glitchtokens中，则将该批次的flag置为1
            if any(token.item() in self.glitchtokens for token in token_ids[b]):
                config_flag[b] = 1
                
        return config_flag.view(-1, 1, 1)
    
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 计算config_flag
        config_flag = self.create_config_flag(input_ids)
        # 存储原始forward方法
        original_forwards = {}
        
        # 修改所有LinearWithLoRA层的forward调用
        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx in target_layers:
                if lora_mlp:
                    # 保存原始forward方法
                    original_forwards[f'gate_proj_{layer_idx}'] = layer.mlp.gate_proj.forward
                    original_forwards[f'up_proj_{layer_idx}'] = layer.mlp.up_proj.forward
                    
                    # 使用functools.partial来正确捕获config_flag
                    from functools import partial
                    layer.mlp.gate_proj.forward = partial(original_forwards[f'gate_proj_{layer_idx}'], config_flag=config_flag)
                    layer.mlp.up_proj.forward = partial(original_forwards[f'up_proj_{layer_idx}'], config_flag=config_flag)
        
        # 调用原始模型的forward
        outputs = self.model(
            input_ids, attention_mask=attention_mask, **kwargs
        )
        
        # 确保输出有logits属性
        if not hasattr(outputs, "logits") and isinstance(outputs, torch.Tensor):
            outputs.logits = outputs

        # 恢复原始forward方法
        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx in target_layers:
                if lora_mlp:
                    layer.mlp.gate_proj.forward = original_forwards[f'gate_proj_{layer_idx}']
                    layer.mlp.up_proj.forward = original_forwards[f'up_proj_{layer_idx}']
        
        return outputs
        
    def generate(self, input_ids,  **kwargs):
        # 计算初始的config_flag
        config_flag = self.create_config_flag(input_ids)
        
        # 存储原始forward方法
        original_forwards = {}
        
        # 修改所有LinearWithLoRA层的forward调用
        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx in target_layers:
                if lora_mlp:
                    # 保存原始forward方法
                    original_forwards[f'gate_proj_{layer_idx}'] = layer.mlp.gate_proj.forward
                    original_forwards[f'up_proj_{layer_idx}'] = layer.mlp.up_proj.forward
                    
                    # 使用functools.partial来正确捕获config_flag
                    from functools import partial
                    layer.mlp.gate_proj.forward = partial(original_forwards[f'gate_proj_{layer_idx}'], config_flag=config_flag)
                    layer.mlp.up_proj.forward = partial(original_forwards[f'up_proj_{layer_idx}'], config_flag=config_flag)
        
        # 调用原始模型的generate
        outputs = self.model.generate(
            input_ids=input_ids,
            **kwargs
        )
        
        # 恢复原始forward方法
        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx in target_layers:
                if lora_mlp:
                    layer.mlp.gate_proj.forward = original_forwards[f'gate_proj_{layer_idx}']
                    layer.mlp.up_proj.forward = original_forwards[f'up_proj_{layer_idx}']
        
        return outputs
        
    
from functools import partial



# 创建LoRA替换函数，包含设备和数据类型信息
assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, 
                        device=model.device, dtype=model.dtype)

# 选择性地替换指定范围内的层
for layer_idx, layer in enumerate(model.model.layers):
    if layer_idx in target_layers:  # 只替换目标范围内的层
        if lora_mlp:
            layer.mlp.gate_proj = assign_lora(layer.mlp.gate_proj)
            layer.mlp.up_proj = assign_lora(layer.mlp.up_proj)

# 创建包装后的模型
wrapped_model = ModelWithConfig(model, glitchtokens,tokenizer)



# 获取需要训练的参数（只训练LoRA参数）
def get_trainable_params(model):
    trainable_params = []
    for name, param in model.named_parameters():
        if "lora" in name and param.requires_grad:
            trainable_params.append(param)
    return trainable_params

# 获取可训练参数
trainable_params = get_trainable_params(wrapped_model)
print(f"可训练参数数量: {len(trainable_params)}")



# 创建优化器
optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

# 计算总的训练步数（考虑梯度累积）
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
num_training_steps = num_epochs * num_update_steps_per_epoch
num_warmup_steps = int(warmup_ratio * num_training_steps)

# 创建学习率调度器
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# 用于记录损失变化
epoch_losses = []
step_losses = []
steps = []

# 训练循环
wrapped_model.train()
progress_bar = tqdm(range(num_training_steps))
completed_steps = 0

for epoch in range(num_epochs):
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_dataloader):
        # 获取输入和标签
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # 前向传播
        outputs = wrapped_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # 计算损失并根据梯度累积步数缩放
        loss = outputs.loss / gradient_accumulation_steps
        total_loss += loss.item() * gradient_accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 只在累积完成后更新参数
        if (batch_idx + 1) % gradient_accumulation_steps == 0 or batch_idx == len(train_dataloader) - 1:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            
            # 优化器步骤
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # 记录每一步的损失
            step_losses.append(loss.item() * gradient_accumulation_steps)
            steps.append(completed_steps)
            
            completed_steps += 1
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1} - Loss: {loss.item() * gradient_accumulation_steps:.4f}")
            
            # 可以添加保存检查点的代码
            if completed_steps % 100 == 0:
                print(f"Step {completed_steps}: Loss = {loss.item() * gradient_accumulation_steps:.4f}")
    
    avg_loss = total_loss / len(train_dataloader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs} 完成，平均损失: {avg_loss:.4f}")

# Plot loss curves
plt.figure(figsize=(12, 5))

# Plot step-wise loss curve
plt.subplot(1, 2, 1)
plt.plot(steps, step_losses)
plt.title('Loss per Step during Training')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.grid(True)

# Plot epoch-wise average loss curve
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), epoch_losses, 'bo-')
plt.title('Average Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.xticks(range(1, num_epochs+1))
plt.grid(True)

plt.tight_layout()
plt.show()

print("训练完成！")
def repetive_judge(tokenizer, token_id,model_path,device):
        # NOTE: 空白字符开头的token如果能以去掉空白字符的形式输出就不算glitchtoken
        token = tokenizer.decode([token_id])
        token_without_space = tokenizer.decode([token_id]).lstrip()
        string_to_repeat = token_without_space
        if 'Yi' in model_path:
#             token = self.tokenizer.decode([token_id])
            content = f"Can you repeat the character '{token}' and return back to me?\nAnswer: Here is the repeated string:\n"
            messages = [{"role": "user", "content": content}]
            input_ids = tokenizer.apply_chat_template(
                    conversation=messages, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    return_tensors='pt'
                ).to(device)
            output_ids = wrapped_model.generate(input_ids,max_new_tokens=10, do_sample=False,temperature=None, top_p=None)[0]
            all_response = tokenizer.decode(output_ids, skip_special_tokens=True)
            k = all_response.find("assistant") + len("assistant")
            response = all_response[k:]
            if token_without_space in response :
                return True
            else:
                # print(all_response)
                return False

        else:
            text1 = "Question: Can you repeat the string '"
            text2 = "' and return back to me?\nAnswer: Here is the repeated string:\n"
            tokens1 = torch.tensor(tokenizer.encode(text1))
            tokens2 = torch.tensor(tokenizer.encode(text2))
            if 'Qw' in model_path: # qwen will not add special token in the begging
                tokens = torch.cat((tokens1, torch.tensor([token_id]), tokens2), dim=0).to(device)
            else:
                tokens = torch.cat((tokens1, torch.tensor([token_id]), tokens2[1:]), dim=0).to(device)
            text = f"Question: Can you repeat the string '{token}' and return back to me?\nAnswer: Here is the repeated string:\n"
            k = len(text)
            tokens = torch.unsqueeze(tokens, dim=0)


            response_tokens = wrapped_model.generate(tokens,max_new_tokens=10, do_sample=False,temperature=None, top_p=None,pad_token_id = tokenizer.eos_token_ids )[0]
            all_response = tokenizer.decode(response_tokens, skip_special_tokens=True) # 解码 token IDs
            response = all_response[k:]
            if string_to_repeat in response:
                return True
            else:
                # print(all_response)
                return False
            
# 统计通过repetive_judge检测的token数量
total_tokens = len(glitchtokens)
passed_tokens = 0
passed_token_ids = []

# 遍历每个glitch token进行检测
for token_id in glitchtokens:
    
    # 调用repetive_judge函数检测
    # if repetive_judge(model,tokenizer, token_id, model_path, device):
    if repetive_judge(tokenizer, token_id, model_path, device):
        passed_tokens += 1
        passed_token_ids.append(token_id)
    
    # 可以添加进度显示
    if (glitchtokens.index(token_id) + 1) % 100 == 0:
        print(f"已通过{passed_tokens}/{glitchtokens.index(token_id) + 1}，{passed_tokens/(glitchtokens.index(token_id) + 1)*100:.2f}%")
        print(f"已处理 {glitchtokens.index(token_id) + 1}/{total_tokens} 个tokens")

# 输出通过检测的token示例（最多显示5个）
if passed_tokens > 0:
    print("通过检测的tokens示例:")
    for i, token_id in enumerate(passed_token_ids[:5]):
        token = tokenizer.decode([token_id])
        print(f"  {i+1}. Token ID: {token_id}, Token: '{token}'")
    
    if passed_tokens > 5:
        print(f"  ... 以及其他 {passed_tokens - 5} 个tokens")

print(f"测试模型为{model_path}")
print(f"训练轮数为{num_epochs}")
print(f"训练批次大小为{batch_size*gradient_accumulation_steps}")
print(f"keylayer为{target_layers}")
print(f"lora_r为{lora_r},lora_缩放因子为{lora_alpha}")
# 输出结果统计
print(f"总共检测了 {total_tokens} 个glitch tokens")
print(f"通过repetive_judge检测的tokens数量: {passed_tokens} ({passed_tokens/total_tokens*100:.2f}%)")

# 创建保存目录
save_dir = '/root/autodl-fs/训练代码/用很小的alpha的实验/结果/mistral'
os.makedirs(save_dir, exist_ok=True)

# 收集结果数据
results_data = {
    'model_path': model_path,
    'num_epochs': num_epochs,
    'batch_size': batch_size * gradient_accumulation_steps,
    'learning_rate': learning_rate,
    'target_layers': str(list(target_layers)),
    'lora_r': lora_r,
    'lora_alpha': lora_alpha,
    'total_tokens': total_tokens,
    'passed_tokens': passed_tokens,
    'fix_rate': passed_tokens/total_tokens * 100
}

# 转换为DataFrame
df = pd.DataFrame([results_data])

# 使用beta、lora_r和lora_alpha构建文件名
filename = f'r_{lora_r}_alpha_{lora_alpha}_fix_rate_results.csv'
file_path = os.path.join(save_dir, filename)

# 保存到CSV
df.to_csv(file_path, index=False)
print(f"修复率结果已保存到: {file_path}")




os.environ["HF_ALLOW_CODE_EVAL"] = "1"



# 忽略特定警告
warnings.filterwarnings("ignore", message="Failed to get model SHA for")

FIXED_SEED = 42 # 选择任何整数作为种子
print(f"{FIXED_SEED}")
# 直接使用HFLM评估包装后的模型
results = evaluator.simple_evaluate(
    model=HFLM(pretrained=wrapped_model, tokenizer=wrapped_model.tokenizer),
    tasks=["mmlu","gsm8k"],  # 或其他您想评估的任务
    verbosity="WARNING",
    batch_size=8,
    # 设置固定的随机种子参数
    random_seed=FIXED_SEED,
    numpy_random_seed=FIXED_SEED,
    torch_random_seed=FIXED_SEED,
    fewshot_random_seed=FIXED_SEED,
)

print(results["results"])

resulte = results["results"]


import os

# Create directory if it doesn't exist
save_dir = '/root/autodl-fs/训练代码/用很小的alpha的实验/评估结果/mistral'
os.makedirs(save_dir, exist_ok=True)

# Convert dictionary to DataFrame
# Extract metrics and their values
metrics = []
values = []
for task, results in resulte.items():
    for metric, value in results.items():
        if isinstance(value, (int, float)):  # Only include numeric values
            metrics.append(f"{task}_{metric}")
            values.append(value)

df = pd.DataFrame({
    'metric': metrics,
    'value': values
})

# Save to CSV
csv_path = os.path.join(save_dir, f'r_{lora_r}_alpha_{lora_alpha}_fix_rate_results.csv')
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")