from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import os

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, device, dtype):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        A_tensor = torch.randn(in_dim, rank, device=device,dtype = dtype) * std_dev
        B_tensor = torch.zeros(rank, out_dim, device=device,dtype = dtype)
        
        self.A = torch.nn.Parameter(A_tensor)
        self.B = torch.nn.Parameter(B_tensor)
        self.alpha = alpha
        self.rank = rank
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
        if config_flag is None:
            return self.linear(x)
        else:
            return self.linear(x) + (config_flag * self.lora(x))
        

class GlitchCleaner(torch.nn.Module):
    def __init__(self, model, glitchtokens, tokenizer, target_layers):
        super().__init__()
        self.model = model
        self.glitchtokens = glitchtokens
        self.tokenizer = tokenizer
        self.device = model.device
        self.config = model.config
        self.tie_weights = lambda: self
        self.target_layers = target_layers

        self.is_qwen = hasattr(model, "transformer") and hasattr(model.transformer, "h")
        self.is_llama = hasattr(model, "model") and hasattr(model.model, "layers")

    def to(self, *args, **kwargs):
        return self.model.to(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def create_config_flag(self, token_ids):
        batch_size = token_ids.shape[0]
        config_flag = torch.zeros(batch_size, device=self.model.device, dtype=self.model.dtype)
        for b in range(batch_size):
            if any(token.item() in self.glitchtokens for token in token_ids[b]):
                config_flag[b] = 1
        return config_flag.view(-1, 1, 1)

    def _patch_lora_forward(self, config_flag):
        original_forwards = {}
        if self.is_qwen:
            for layer_idx, layer in enumerate(self.model.transformer.h):
                if layer_idx in self.target_layers:
                        original_forwards[f'w1_{layer_idx}'] = layer.mlp.w1.forward
                        original_forwards[f'w2_{layer_idx}'] = layer.mlp.w2.forward
                        from functools import partial
                        layer.mlp.w1.forward = partial(original_forwards[f'w1_{layer_idx}'], config_flag=config_flag)
                        layer.mlp.w2.forward = partial(original_forwards[f'w2_{layer_idx}'], config_flag=config_flag)
        elif self.is_llama:
            for layer_idx, layer in enumerate(self.model.model.layers):
                if layer_idx in self.target_layers:
                        original_forwards[f'gate_proj_{layer_idx}'] = layer.mlp.gate_proj.forward
                        original_forwards[f'up_proj_{layer_idx}'] = layer.mlp.up_proj.forward
                        from functools import partial
                        layer.mlp.gate_proj.forward = partial(original_forwards[f'gate_proj_{layer_idx}'], config_flag=config_flag)
                        layer.mlp.up_proj.forward = partial(original_forwards[f'up_proj_{layer_idx}'], config_flag=config_flag)
        return original_forwards

    def _restore_lora_forward(self, original_forwards):
        if self.is_qwen:
            for layer_idx, layer in enumerate(self.model.transformer.h):
                if layer_idx in self.target_layers:
                        layer.mlp.w1.forward = original_forwards[f'w1_{layer_idx}']
                        layer.mlp.w2.forward = original_forwards[f'w2_{layer_idx}']
        elif self.is_llama:
            for layer_idx, layer in enumerate(self.model.model.layers):
                if layer_idx in self.target_layers:
                        layer.mlp.gate_proj.forward = original_forwards[f'gate_proj_{layer_idx}']
                        layer.mlp.up_proj.forward = original_forwards[f'up_proj_{layer_idx}']

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        config_flag = self.create_config_flag(input_ids)
        original_forwards = self._patch_lora_forward(config_flag)
        outputs = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        if not hasattr(outputs, "logits") and isinstance(outputs, torch.Tensor):
            outputs.logits = outputs
        self._restore_lora_forward(original_forwards)
        return outputs

    def generate(self, input_ids, **kwargs):
        if isinstance(input_ids, str):
            input_ids = self.tokenizer(input_ids, return_tensors="pt").input_ids.to(self.device)
        elif isinstance(input_ids, torch.Tensor):
            input_ids = input_ids
        else:
            raise ValueError("Input must be a text string or a tensor of token_ids")
        config_flag = self.create_config_flag(input_ids)
        original_forwards = self._patch_lora_forward(config_flag)
        outputs = self.model.generate(input_ids=input_ids, **kwargs)
        self._restore_lora_forward(original_forwards)
        return outputs
    

def load_lora_parameters(model_path):
    
    if 'Qwen-7B-Chat' in model_path:
        hf_model_path = "/root/autodl-tmp/Qwen-7B-Chat"
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path,trust_remote_code=True)
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token = '<|endoftext|>'
            tokenizer.eos_token_id = 151643 
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        if 'Llama-2-7b-chat' in model_path:
            hf_model_path = "meta-llama/Llama-2-7b-chat-hf"
        elif 'Mistral-7B-Instruct-v0.1' in model_path:        
            hf_model_path = "mistralai/Mistral-7B-Instruct-v0.1"
        elif 'gemma-2b-it' in model_path:
            hf_model_path = "google/gemma-2b-it"
        elif 'Yi-6B-Chat' in model_path:
            hf_model_path = "01-ai/Yi-6B-Chat"
        elif 'Deepseek-llm-7b-chat' in model_path:
            hf_model_path = "deepseek-ai/deepseek-llm-7b-chat"
        else:   
            raise ValueError(f"Unsupported model path: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_path,
            device_map="auto",
        )
        
    model.requires_grad_(False)
    lora_path = os.path.join("LoRA-Parameter", f"{model_path}.pt")
    df = pd.read_csv('Glitchtokens/' + model_path + '-glitch-tokens.csv')
    glitchtokens = df['index'].tolist()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("设置pad_token为eos_token")
    else:
        print(f"已存在pad_token: {tokenizer.pad_token}")

    

    lora_params = torch.load(lora_path, map_location='cpu')
    config = lora_params.get('config', {})
    lora_r = config.get('lora_r', 4)
    lora_alpha = config.get('lora_alpha', lora_r)
    target_layers = config.get('target_layers', range(19, 29))

    from functools import partial
    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, 
                        device=model.device, dtype=model.dtype)

    if "qwen" in model_path.lower():
        for layer_idx, layer in enumerate(model.transformer.h):
            if layer_idx in target_layers:
                layer.mlp.w1 = assign_lora(layer.mlp.w1)
                layer.mlp.w2 = assign_lora(layer.mlp.w2)
        for name, param in lora_params.items():
            if name == 'config':
                continue
            parts = name.split('_')
            if len(parts) < 3:
                continue
            module_type = parts[0]  # w1 或 w2
            layer_idx = int(parts[1])
            param_type = parts[2]  # A 或 B
            if layer_idx in target_layers:
                if module_type == 'w1':
                    if param_type == 'A':
                        model.transformer.h[layer_idx].mlp.w1.lora.A.data.copy_(param.to(model.device))
                    elif param_type == 'B':
                        model.transformer.h[layer_idx].mlp.w1.lora.B.data.copy_(param.to(model.device))
                elif module_type == 'w2':
                    if param_type == 'A':
                        model.transformer.h[layer_idx].mlp.w2.lora.A.data.copy_(param.to(model.device))
                    elif param_type == 'B':
                        model.transformer.h[layer_idx].mlp.w2.lora.B.data.copy_(param.to(model.device))
    else:
        for layer_idx, layer in enumerate(model.model.layers):
            if layer_idx in target_layers:
                layer.mlp.gate_proj = assign_lora(layer.mlp.gate_proj)
                layer.mlp.up_proj = assign_lora(layer.mlp.up_proj)
        for name, param in lora_params.items():
            if name == 'config':
                continue
            parts = name.split('_')
            if len(parts) < 4:
                continue
            module_type = parts[0] + '_' + parts[1]  
            layer_idx = int(parts[2])
            param_type = parts[3]  
            if layer_idx in target_layers:
                if module_type == 'gate_proj':
                    if param_type == 'A':
                        model.model.layers[layer_idx].mlp.gate_proj.lora.A.data.copy_(param.to(model.device))
                    elif param_type == 'B':
                        model.model.layers[layer_idx].mlp.gate_proj.lora.B.data.copy_(param.to(model.device))
                elif module_type == 'up_proj':
                    if param_type == 'A':
                        model.model.layers[layer_idx].mlp.up_proj.lora.A.data.copy_(param.to(model.device))
                    elif param_type == 'B':
                        model.model.layers[layer_idx].mlp.up_proj.lora.B.data.copy_(param.to(model.device))

    if glitchtokens is not None:
        GlitchCleaner_modle = GlitchCleaner(model, glitchtokens, tokenizer,target_layers)
        return GlitchCleaner_modle
    else:
        return model
    
def count_passed_glitchtokens(model,model_path):
    
    tokenizer = model.tokenizer
    device = model.device
    glitchtokens = model.glitchtokens
    def repetive_judge(token_id):
        token = tokenizer.decode([token_id])
        token_without_space = token.lstrip()
        string_to_repeat = token_without_space
        if 'Yi' in model_path:
            content = f"Can you repeat the character '{token}' and return back to me?\nAnswer: Here is the repeated string:\n"
            messages = [{"role": "user", "content": content}]
            input_ids = tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt'
            ).to(device)
            prompt_len = input_ids.shape[1]

            output_ids = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                temperature=None,
                top_p=None,
            )[0]

            gen_ids = output_ids[prompt_len:]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True)

            return token_without_space in response
        else:
            text1 = "Question: Can you repeat the string '"
            text2 = "' and return back to me?\nAnswer: Here is the repeated string:\n"
            tokens1 = torch.tensor(tokenizer.encode(text1))
            tokens2 = torch.tensor(tokenizer.encode(text2))
            if 'Qw' in model_path:  # qwen will not add special token in the beginning
                tokens = torch.cat((tokens1, torch.tensor([token_id]), tokens2), dim=0).to(device)
            else:
                tokens = torch.cat((tokens1, torch.tensor([token_id]), tokens2[1:]), dim=0).to(device)
            text = f"Question: Can you repeat the string '{token}' and return back to me?\nAnswer: Here is the repeated string:\n"
            k = len(text)
            tokens = torch.unsqueeze(tokens, dim=0)
            response_tokens = model.generate(tokens, max_new_tokens=10, do_sample=False, temperature=None, top_p=None)[0]
            all_response = tokenizer.decode(response_tokens, skip_special_tokens=True)
            response = all_response[k:]
            return string_to_repeat in response

    total_tokens = len(glitchtokens)
    passed_tokens = 0
    passed_token_ids = []

    for idx, token_id in enumerate(glitchtokens):
        if repetive_judge(token_id):
            passed_tokens += 1
            passed_token_ids.append(token_id)

        if (idx + 1) % 100 == 0:
            print(f"Passed {passed_tokens}/{idx + 1}, {passed_tokens/(idx + 1)*100:.2f}%")
            print(f"Processed {idx + 1}/{total_tokens} tokens")