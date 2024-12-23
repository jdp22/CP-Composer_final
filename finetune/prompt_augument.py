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
tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side='left', truncation_side='left')

def batch_generate_responses(prompts):
    """Generate responses for a batch of prompts."""
    messages = [
        {
            "role": "system",
            "content": (
            "Rephrase the given sentence into another English sentence with the same meaning. Ensure the output varies in phrasing, structure, or vocabulary while staying as a single line. Do not include anthing else than the rephrased sentence. The output should maintain the original context and meaning while introducing variations in phrasing or vocabulary."
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
    model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    # Generate responses
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
    
    # Decode outputs
    decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    responses = [
        response.split("\nassistant\n", 1)[-1].strip() if "\nassistant\n" in response else response.strip()
        for response in decoded_responses
    ]
    return responses

def process_batch(lines):
    """Process a batch of lines."""
    processed_lines = []
    prompts = []
    original_parts = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) > 1:
            last_prompt = parts[-1]
            if random.random() < 0.5:
                amoid_sequence = parts[10]
                max_positions = len(amoid_sequence) // 2
                num_positions = random.randint(1, max_positions)
                positions = random.sample(range(len(amoid_sequence)), num_positions)
                for i in positions:
                    last_prompt += ' ' + f'The amino acid at position {i} is {amino_acid_map[amoid_sequence[i]]}.'
            prompts.append(last_prompt)
            original_parts.append(parts)
        else:
            processed_lines.append(line.strip())  # Keep lines without tabs unchanged

    # Generate responses in batch
    if prompts:
        responses = batch_generate_responses(prompts)
        for parts, response in zip(original_parts, responses):
            parts[-1] = response
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
file_path = "/data/private/jdp/PepGLAD/datasets/ProtFrag/processed/prompt_distance_index.txt"  # Replace with the path to your text file
process_file(file_path, batch_size=256)

