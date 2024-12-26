import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
import random

# Amino acid map
amino_acid_map = {
    'A': 'Alanine', 'R': 'Arginine', 'N': 'Asparagine', 'D': 'Aspartic acid',
    'C': 'Cysteine', 'E': 'Glutamic acid', 'Q': 'Glutamine', 'G': 'Glycine',
    'H': 'Histidine', 'I': 'Isoleucine', 'L': 'Leucine', 'K': 'Lysine',
    'M': 'Methionine', 'F': 'Phenylalanine', 'P': 'Proline', 'S': 'Serine',
    'T': 'Threonine', 'W': 'Tryptophan', 'Y': 'Tyrosine', 'V': 'Valine'
}

# Set up Hugging Face's API endpoint mirror
cache_dir = '/data/private/jdp/Qwen2.5-7B-Instruct'
model_name = "Qwen/Qwen2.5-7B-Instruct"
print(f'Model name is {model_name}')

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    cache_dir=cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left', truncation_side='left')

def batch_generate_responses(prompts):
    """Generate responses for a batch of prompts."""
    messages = [
        {
            "role": "system",
            "content": (
                "Rephrase the given sentence into another English sentence with the same meaning. "
                "Ensure the output varies in phrasing, structure, or vocabulary while staying as a "
                "single line. Do not include anything else than the rephrased sentence. "
                "The output should maintain the original context and meaning while introducing "
                "variations in phrasing or vocabulary."
            )
        }
    ]
    
    # Prepare batch texts
    batch_texts = [
        tokenizer.apply_chat_template(
            messages + [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            cache_dir=cache_dir
        )
        for prompt in prompts
    ]
    
    # Tokenize inputs
    model_inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)
    
    # Generate responses
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
    
    # Decode outputs
    decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    responses = []
    for response in decoded_responses:
        if "\nassistant\n" in response:
            # 分隔符可能因模型不同而异，这里根据你实际情况做适当处理
            content = response.split("\nassistant\n", 1)[-1].strip()
        else:
            content = response.strip()
        responses.append(content)
    return responses

def process_batch(lines):
    """Process a batch of lines."""
    processed_lines = []
    prompts1 = []
    prompts2 = []
    original_parts = []

    for line in lines:
        parts = line.strip().split('\t')
        last_prompt = parts[-1]
        prompts1.append(last_prompt)

        amoid_sequence = parts[10]
        num_positions = 2
        positions = random.sample(range(len(amoid_sequence)), num_positions)
        # 生成 prompts2 的内容
        amino_prompt = ''
        for i in positions:
            amino_prompt += f'The amino acid at position {i} is {amino_acid_map[amoid_sequence[i]]}.'
        prompts2.append(amino_prompt)

        original_parts.append(parts)

    # **修改点：合并 prompts1 和 prompts2，一次性调用 batch_generate_responses**
    combined_prompts = prompts1 + prompts2
    combined_responses = batch_generate_responses(combined_prompts)

    # 前半部分对应 prompts1，后半部分对应 prompts2
    responses1 = combined_responses[:len(prompts1)]
    responses2 = combined_responses[len(prompts1):]

    # 将 results 填充回去
    for parts, response1, response2 in zip(original_parts, responses1, responses2):
        parts[-1] = response1  # 更新最后一列
        parts.append(response2)  # 追加新的列
        processed_lines.append('\t'.join(parts))
    return processed_lines

def process_file(file_path, batch_size=16):
    """Read the file, update the last prompt in each line using batch processing, and save the changes."""
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist!")
        return

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    updated_lines = []
    for i in tqdm(range(0, len(lines), batch_size), desc="Processing batches"):
        batch_lines = lines[i:i + batch_size]
        updated_lines.extend(process_batch(batch_lines))

    # Write the updated lines to a new file with "augmentation" added to the name
    new_file_path = file_path.replace(".txt", "_augmentation.txt")
    with open(new_file_path, 'w', encoding='utf-8') as file:
        for updated_line in updated_lines:
            file.write(updated_line + '\n')

    print(f"File {new_file_path} has been created with updated content!")

# Example usage
if __name__ == "__main__":
    file_path = "/data/private/jdp/PepGLAD/datasets/train_valid/processed/prompt_valid_distance_index.txt"  # Replace with the path to your text file
    process_file(file_path, batch_size=256)


