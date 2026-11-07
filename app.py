import streamlit as st
from transformers import pipeline

# ====== CHANGE THIS to your own Hugging Face model repo (from the Colab training) ======
GPT_MODEL_ID = "YOUR_HF_USERNAME/balochi-gpt2"
BERT_MODEL_ID = "AnasAlokla/multilingual_go_emotions"
# ========================================================================================

st.set_page_config(page_title="BERT vs GPT", page_icon="🤖", layout="centered")

# Only ONE model is kept in memory at a time (keeps the free Streamlit server from running out of RAM)
@st.cache_resource(max_entries=1, show_spinner="Loading model... (first time takes a minute)")
def load_model(kind):
    if kind == "bert":
        return pipeline("text-classification", model=BERT_MODEL_ID, top_k=None)
    return pipeline("text-generation", model=GPT_MODEL_ID)

st.title("🤖 BERT vs GPT: Encoder vs Decoder")
st.write("Choose a section from the left menu.")

section = st.sidebar.radio(
    "Sections",
    ["Section 1: BERT (Encoder-only)", "Section 2: GPT (Decoder-only)"],
)

# ------------------------------------------------------------------ SECTION 1
if section.startswith("Section 1"):
    st.header("🎭 Section 1: Multilingual Emotion Detection")
    st.write(
        "**Model:** BERT (encoder-only) | **Task:** emotion detection | "
        "**Languages:** high-resource, multilingual (English, Spanish, French, Arabic, Dutch, Turkish)"
    )

    examples = {
        "English": "I am so happy today, everything is going great!",
        "Spanish": "Estoy muy triste porque perdí mi trabajo.",
        "French": "Je suis tellement en colère contre cette décision.",
        "Arabic": "أنا ممتن جدا لمساعدتك",
    }
    choice = st.selectbox("Pick an example or write your own below:", ["(write my own)"] + list(examples))
    text = st.text_area("Enter a short sentence:", value=examples.get(choice, ""), height=100, key="bert_text")

    if st.button("Detect emotion"):
        if not text.strip():
            st.warning("Please type a sentence first.")
        else:
            classifier = load_model("bert")
            result = classifier(text)
            scores = result[0] if isinstance(result[0], list) else result
            scores = sorted(scores, key=lambda x: x["score"], reverse=True)

            st.subheader("Input")
            st.write(text)
            st.subheader("Predicted emotion")
            st.success(f"{scores[0]['label'].upper()}  ({scores[0]['score']:.1%})")
            st.subheader("Top 5 emotions")
            st.bar_chart({s["label"]: s["score"] for s in scores[:5]})

    st.caption("BERT reads the whole sentence at once (both directions) to understand it. It does not generate text.")

# ------------------------------------------------------------------ SECTION 2
else:
    st.markdown(
        "<style>.rtl {direction: rtl; text-align: right; font-size: 20px;}</style>",
        unsafe_allow_html=True,
    )
    st.header("✍️ Section 2: Balochi Text Generation")
    st.write(
        "**Model:** GPT-2 (decoder-only) | **Task:** text generation | "
        "**Language:** Balochi (low-resource, monolingual)"
    )

    prompt = st.text_area("Write the beginning of a Balochi sentence:", height=100, key="gpt_text")
    max_new = st.slider("How many new tokens to generate", 10, 100, 40)
    temperature = st.slider("Creativity (temperature)", 0.5, 1.5, 0.9)

    if st.button("Generate"):
        if not prompt.strip():
            st.warning("Please type the beginning of a sentence.")
        else:
            generator = load_model("gpt")
            out = generator(
                prompt,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                repetition_penalty=1.2,
            )
            st.subheader("Input")
            st.markdown(f"<div class='rtl'>{prompt}</div>", unsafe_allow_html=True)
            st.subheader("Generated text")
            st.markdown(f"<div class='rtl'>{out[0]['generated_text']}</div>", unsafe_allow_html=True)

    st.caption("GPT-2 predicts the next token again and again, using only the previous tokens (left to right).")
