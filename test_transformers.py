from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("✅ AutoTokenizer loaded successfully!")
print(tokenizer)
