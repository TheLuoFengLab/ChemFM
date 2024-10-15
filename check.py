from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("feiyang-cai/ChemFM-1B")
tokenizer = AutoTokenizer.from_pretrained("feiyang-cai/ChemFM-1B")

print(model)
print(tokenizer)