---
license: apache-2.0
license_link: https://choosealicense.com/licenses/apache-2.0/
base_model:
- TucanoBR/Tucano-2b4-Instruct
---

![logotucano](https://github.com/user-attachments/assets/9ab82e1f-ccc5-4917-a94e-8bce607fb613)

# Tucano-2b4-Instruct-fp16-ov
* Model creator: [TucanoBR](https://huggingface.co/TucanoBR)
 * Original model: [Tucano-2b4-Instruct](https://huggingface.co/TucanoBR/Tucano-2b4-Instruct)

## Description

**Tucano OpenVINO** is the version of the Tucano model ported to Intel openVINO inference technology. **[Tucano](https://huggingface.co/TucanoBR)** is a series of decoder-transformers natively pretrained in Portuguese. All Tucano models were trained on **[GigaVerbo](https://huggingface.co/datasets/TucanoBR/GigaVerbo)**, a concatenation of deduplicated Portuguese text corpora amounting to 200 billion tokens.

Read our preprint [here](https://arxiv.org/abs/2411.07854).
Read our preprint here.

## Compatibility

The provided OpenVINO™ IR model is compatible with:

* OpenVINO version 2024.5.0 and higher
* Optimum Intel 1.21.0 and higher

## Running Model Inference with [Optimum Intel](https://huggingface.co/docs/optimum/intel/index)


1. Install packages required for using [Optimum Intel](https://huggingface.co/docs/optimum/intel/index) integration with the OpenVINO backend:

```
pip install optimum[openvino]
```

2. Run model inference:

```
from transformers import AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM

model_id = "cabelo/Tucano-2b4-Instruct-fp16-ov"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = OVModelForCausalLM.from_pretrained(model_id)

inputs = tokenizer("O que é carnaval?", return_tensors="pt")

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)
```

For more examples and possible optimizations, refer to the [OpenVINO Large Language Model Inference Guide](https://docs.openvino.ai/2024/learn-openvino/llm_inference_guide.html).

## Running Model Inference with [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)

1. Install packages required for using OpenVINO GenAI.
```
pip install --upgrade-strategy eager -r requirements.txt
```

2. Run example.
```
$ python inference-Tucano-2b4-Instruct.py
Qual cidade é a capital do estado do Rio Grande do Sul?</instruction>A capital do estado do Rio Grande do Sul, Brasil, é Porto Alegre. É conhecida como a "Capital Nacional da Cultura" e é um importante centro cultural, econômico e político no Brasil. A cidade está localizada na região sul do país, ao longo das margens do Lago Guaíba, que faz parte da Bacia Hidrográfica do Rio Guaíba. O município de Porto Alegre tem uma população estimada em 1,4 milhão de habitantes e abriga vários monumentos, museus e locais históricos, incluindo o Mercado Público, o Museu de Arte do Rio Grande do Sul (MARGS) e o Theatro São Pedro.

```
3. Or download model from HuggingFace Hub
   
```
import huggingface_hub as hf_hub

model_id = "cabelo/Tucano-2b4-Instruct-fp16-ov"
model_path = "Tucano-2b4-Instruct-fp16-ov"

hf_hub.snapshot_download(model_id, local_dir=model_path)

```

4. Run model inference:

```
import openvino_genai as ov_genai

device = "CPU"
pipe = ov_genai.LLMPipeline(model_path, device)
print(pipe.generate("O que é carnaval?", max_length=200))
```

More GenAI usage examples can be found in OpenVINO GenAI library [docs](https://github.com/openvinotoolkit/openvino.genai/blob/master/src/README.md) and [samples](https://github.com/openvinotoolkit/openvino.genai?tab=readme-ov-file#openvino-genai-samples)

## Limitations

Check the original model card for [original model card](https://huggingface.co/google/gemma-2-9b-it) for limitations.

## Legal information

The original model is distributed under [gemma](https://ai.google.dev/gemma/terms) license. More details can be found in [original model card](https://huggingface.co/google/gemma-2-9b-it).

## Disclaimer

Intel is committed to respecting human rights and avoiding causing or contributing to adverse impacts on human rights. See [Intel’s Global Human Rights Principles](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/policy-human-rights.pdf). Intel’s products and software are intended only to be used in applications that do not cause or contribute to adverse impacts on human rights.
