import streamlit as st
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

st.set_page_config(
    page_title="Financial Sentiment Analyzer",
    layout="centered"
)

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
ADAPTER_ID = "Yutakuroda/llama-3.2-1b-fingpt-sentiment"

@st.cache_resource(show_spinner="Loading model weights into memory...")
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

st.title("Financial Sentiment Analyzer")

# Sample buttons
sample_news = [
    "Tesla beats quarterly revenue expectations as EV deliveries hit record highs.",
    "Company reports wider quarterly net loss amid soaring supply chain costs.",
    "Federal Reserve holds benchmark interest rate steady as expected."
]

st.write("**Try a sample headline:**")
cols = st.columns(len(sample_news))
for i, sample in enumerate(sample_news):
    if cols[i].button(f"Headline {i+1}", use_container_width=True):
        st.session_state["news_input"] = sample

user_text = st.text_area(
    "Enter financial headline or market news:",
    value=st.session_state.get("news_input", ""),
    height=120,
    placeholder="Type financial news here..."
)

if st.button("Analyze Sentiment", type="primary", use_container_width=True):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):
            prompt = f"""Instruction:
What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}.

Input:
{user_text.strip()}

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
            raw_out = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()
            clean_out = raw_out.split("Instruction")[0].strip()
            match = re.search(r'\b(positive|negative|neutral)\b', clean_out, re.IGNORECASE)
            sentiment = match.group(0).lower() if match else clean_out.lower()

            st.divider()
            st.subheader("Analysis Result")
            if sentiment == "positive":
                st.success("**Sentiment: POSITIVE**")
            elif sentiment == "negative":
                st.error("**Sentiment: NEGATIVE**")
            elif sentiment == "neutral":
                st.info("**Sentiment: NEUTRAL**")
            else:
                st.warning(f"Result: {sentiment}")