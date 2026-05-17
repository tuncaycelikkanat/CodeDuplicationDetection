import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codes = ["int main() { " + " ".join(["printf('hello');"] * 1000) + " return 0; }", "int a = 1;"]

encoded = tokenizer(
    codes,
    max_length=512,
    truncation=True,
    padding=True,
    return_overflowing_tokens=True,
    return_tensors="pt"
)

print("input_ids shape:", encoded["input_ids"].shape)
print("overflow mapping:", encoded.get("overflow_to_sample_mapping"))
