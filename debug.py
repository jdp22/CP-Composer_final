from transformers import AutoModelForCausalLM, AutoTokenizer
import os
# 设置 Hugging Face 的 API 端点镜像
cache_dir = '/data/private/jdp/Qwen2.5-7B-Instruct'
model_name = "Qwen/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    cache_dir = cache_dir
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompts = "The peptide has two alpha helices. The amino acid at position 3 is Serine."
messages = [
    {"role": "system", "content": "You are an advanced assistant focused on data augmentation for (graph, text) pairs. Your task is to rephrase the given sentence into another sentence in English with the same meaning. You can changing the word order, altering the sentence structure or sequence, or adding, changing or removing words. The output should maintain the original context and meaning while introducing variations in phrasing or vocabulary."},
    {"role": "user", "content": prompts}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    cache_dir = cache_dir
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)