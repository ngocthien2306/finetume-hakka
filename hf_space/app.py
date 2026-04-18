"""
Hakka ↔ Mandarin Translation Demo
HuggingFace Space — runs ngocthien/hakka-translation-model
"""
import os
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "ngocthien/hakka-translation-model"

SYSTEM = {
    "h2m": (
        "你是專業翻譯助手。"
        "請將使用者輸入的客語漢字翻成自然、流暢、忠實原意的繁體中文。"
        "只輸出翻譯結果，不要加解釋、不要加前後綴。"
    ),
    "m2h_sixian": (
        "你是專業翻譯助手。"
        "請將使用者輸入的繁體中文翻成客語漢字（四縣腔）。"
        "只輸出翻譯結果，不要加解釋、不要加前後綴。"
    ),
    "m2h_hailu": (
        "你是專業翻譯助手。"
        "請將使用者輸入的繁體中文翻成客語漢字（海陸腔）。"
        "只輸出翻譯結果，不要加解釋、不要加前後綴。"
    ),
    "chat": (
        "你是專業的客語翻譯助手，精通客語漢字（四縣腔、海陸腔）與繁體中文的互譯。"
        "請根據使用者的指示進行翻譯，只輸出翻譯結果。"
    ),
}

MODE_LABELS = {
    "客語 → 中文 (Hakka → Mandarin)":           "h2m",
    "中文 → 客語四縣腔 (Mandarin → Sixian)":    "m2h_sixian",
    "中文 → 客語海陸腔 (Mandarin → Hailu)":     "m2h_hailu",
}

# ── Load model (once, at startup) ────────────────────────────────────────────
print(f"Loading {MODEL_ID} ...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()
print("Model ready.")


# ── Inference ─────────────────────────────────────────────────────────────────
def _generate(system_prompt: str, user_text: str, max_new_tokens: int = 128) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_text},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()


# ── Tab 1: Structured translation ────────────────────────────────────────────
def translate(text: str, mode_label: str) -> str:
    if not text.strip():
        return ""
    mode = MODE_LABELS[mode_label]
    return _generate(SYSTEM[mode], text)


# ── Tab 2: Free-form chat ─────────────────────────────────────────────────────
def chat(instruction: str) -> str:
    if not instruction.strip():
        return ""
    return _generate(SYSTEM["chat"], instruction)


# ── Gradio UI ─────────────────────────────────────────────────────────────────
EXAMPLES_TRANSLATE = [
    ["逐擺選舉候選人都互相醜化對手，實在分人看毋落去。",   "客語 → 中文 (Hakka → Mandarin)"],
    ["用棉布做个衫褲，𠲿汗又耐著。",                      "客語 → 中文 (Hakka → Mandarin)"],
    ["萬事起頭難，肯努力去做就不用怕做不好。",              "中文 → 客語四縣腔 (Mandarin → Sixian)"],
    ["弟弟在學校擔任班長。",                               "中文 → 客語四縣腔 (Mandarin → Sixian)"],
    ["西洋有一位哥倫布，他發現了新大陸。",                  "中文 → 客語海陸腔 (Mandarin → Hailu)"],
]

EXAMPLES_CHAT = [
    ["幫我把這句話翻成客語四縣腔：西洋有一位哥倫布，他發現了新大陸。"],
    ["這句客語是什麼意思？佢存心愛嚇人。"],
    ["請翻譯成海陸腔：弟弟在學校擔任班長。"],
    ["把這段文字翻成繁體中文：逐擺選舉候選人都互相醜化對手，實在分人看毋落去。"],
]

with gr.Blocks(title="Hakka ↔ Mandarin Translation", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🈶 Hakka ↔ Mandarin Translation
        Fine-tuned **Llama-3-Taiwan-8B-Instruct** · LoRA · Supports **Sixian (四縣腔)** and **Hailu (海陸腔)**
        Model: [ngocthien/hakka-translation-model](https://huggingface.co/ngocthien/hakka-translation-model)
        """
    )

    with gr.Tabs():
        # ── Tab 1 ──
        with gr.Tab("翻譯 Translation"):
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="輸入文字 Input",
                        placeholder="輸入客語或中文...",
                        lines=4,
                    )
                    mode_radio = gr.Radio(
                        choices=list(MODE_LABELS.keys()),
                        value="客語 → 中文 (Hakka → Mandarin)",
                        label="翻譯方向 Direction",
                    )
                    translate_btn = gr.Button("翻譯 Translate", variant="primary")

                with gr.Column():
                    output_text = gr.Textbox(
                        label="翻譯結果 Output",
                        lines=4,
                        interactive=False,
                    )

            gr.Examples(
                examples=EXAMPLES_TRANSLATE,
                inputs=[input_text, mode_radio],
                label="範例 Examples",
            )
            translate_btn.click(
                fn=translate,
                inputs=[input_text, mode_radio],
                outputs=output_text,
            )

        # ── Tab 2 ──
        with gr.Tab("自由問答 Free Chat"):
            gr.Markdown("直接用中文描述翻譯需求，不需要指定 mode。")
            with gr.Row():
                with gr.Column():
                    chat_input = gr.Textbox(
                        label="指令 Instruction",
                        placeholder="例：幫我把這句話翻成客語四縣腔：...",
                        lines=4,
                    )
                    chat_btn = gr.Button("送出 Submit", variant="primary")
                with gr.Column():
                    chat_output = gr.Textbox(
                        label="回應 Response",
                        lines=4,
                        interactive=False,
                    )
            gr.Examples(
                examples=EXAMPLES_CHAT,
                inputs=[chat_input],
                label="範例 Examples",
            )
            chat_btn.click(
                fn=chat,
                inputs=[chat_input],
                outputs=chat_output,
            )

    gr.Markdown(
        """
        ---
        **資料來源 Data:** MOE Hakka Dictionary · HAT Lexicon · Medical Corpus
        **評估結果 Eval:** Corpus BLEU 61.75% (h2m) · 58.95% (zh2hakka) · BERTScore 92.3%
        """
    )

demo.launch()
