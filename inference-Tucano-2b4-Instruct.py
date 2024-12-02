from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

model_id = "cabelo/Tucano-2b4-Instruct-fp16-ov"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = OVModelForCausalLM.from_pretrained(model_id)

inputs = tokenizer("Como é Carnaval no Brasil?", return_tensors="pt")

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
