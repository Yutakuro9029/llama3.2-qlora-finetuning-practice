# Llama-3.2-1B Financial Sentiment Analysis (QLoRA Fine-Tuning)

A hands-on practice repository demonstrating parameter-efficient fine-tuning (**PEFT / QLoRA**) on **Llama-3.2-1B** using **TRL** and **BitsAndBytes (4-bit)** for financial sentiment classification.

**Hugging Face Model Adapter:** [Yutakuroda/llama-3.2-1b-fingpt-sentiment](https://huggingface.co/Yutakuroda/llama-3.2-1b-fingpt-sentiment)

---

## Project Overview

Fine-tuning open-weights Large Language Models (LLMs) on domain-specific classification tasks efficiently on consumer hardware. This experiment trains Meta's `Llama-3.2-1B` on the `FinGPT/fingpt-sentiment-train` dataset to classify financial news and market headlines into sentiment categories (`positive`, `neutral`, `negative`).

### Key Highlights:

- **Base Model:** `meta-llama/Llama-3.2-1B`
- **Fine-Tuning Method:** QLoRA (4-bit NormalFloat quantization via `bitsandbytes`)
- **Dataset:** `FinGPT/fingpt-sentiment-train` (~76.8k samples)
- **Trainable Parameters:** ~11.27M / 1.25B (~0.90% of total weights)
- **Hardware:** Local NVIDIA GPU (RTX 3080 12GB VRAM)

---

## Training Configuration & Results

| Hyperparameter              | Value                                                                       |
| :-------------------------- | :-------------------------------------------------------------------------- |
| **Quantization**            | 4-bit NF4 (`bnb_4bit_compute_dtype=bfloat16`)                               |
| **LoRA Rank ($r$)**         | 16                                                                          |
| **LoRA Alpha ($\alpha$)**   | 32                                                                          |
| **Target Modules**          | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Batch Size (Per Device)** | 2                                                                           |
| **Gradient Accumulation**   | 8 (Effective batch size = 16)                                               |
| **Learning Rate**           | `2e-4`                                                                      |
| **Epochs**                  | 1.0 (4,799 steps)                                                           |
| **Final Loss**              | **~1.02**                                                                   |
| **Mean Token Accuracy**     | **~85.3%**                                                                  |

---

## Project Structure

```text
├── lora_qlora_finetuning.ipynb   # Main Jupyter notebook containing workflow
├── requirements.txt              # Required Python packages
├── .gitignore                    # Ignored model checkpoints and cache
└── README.md                     # Project documentation
```

## Setup & Installation

1. **Clone the repository:**

```markdown
git clone https://github.com/Yutakuro9029/llama3.2-qlora-finetuning-practice.git
cd llama3.2-qlora-finetuning-practice
```

2. **Install dependencies:**

```
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory:

```env
HF_TOKEN=your_huggingface_read_token
WRITE_TOKEN=your_huggingface_write_token
```

---

## Quick Inference

You can run inference directly using the trained LoRA adapter hosted on Hugging Face:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "meta-llama/Llama-3.2-1B"
adapter_name = "Yutakuroda/llama-3.2-1b-fingpt-sentiment"

# 1. Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained(adapter_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. Attach LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_name)
model.eval()

# 3. Test prompt
prompt = """Instruction:
What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.

Input:
Tesla beats quarterly revenue expectations as electric vehicle deliveries hit record highs.

Answer:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=15,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

prompt_len = inputs["input_ids"].shape[1]
raw_output = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
prediction = raw_output.split("Instruction")[0].strip()

print(f"Predicted Sentiment: {prediction}")
# Output: positive

```

---

## Dataset Credit

Training data: `FinGPT/fingpt-sentiment-train` dataset, sourced from Hugging Face Hub (developed by the AI4Finance Foundation), licensed under [MIT License].

- **License:** MIT License — free for both academic and commercial use, provided proper attribution is maintained.
- **Modifications made for this project:** The dataset records (`instruction`, `input`, `output`) were formatted into structured instruction-tuning prompts with EOS tokens for causal language modeling; original text entries and label mappings were not altered.

---

## License

Code in this repository is licensed under the **MIT License**. This license applies to the code and training scripts only:

- The base model weights (`meta-llama/Llama-3.2-1B`) are subject to the **Llama 3.2 Community License Agreement** by Meta.
- The training dataset (`FinGPT/fingpt-sentiment-train`) is separately licensed under the **MIT License** as noted above.
