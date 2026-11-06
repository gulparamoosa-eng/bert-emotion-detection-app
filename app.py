import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Multilingual Emotion Detection (BERT)", page_icon="🎭")

st.title("🎭 Multilingual Emotion Detection")
st.write(
    "**Model type:** BERT (encoder-only) | **Task:** emotion detection | "
    "**Languages:** high-resource, multilingual (English, Spanish, French, Arabic, etc.)"
)

@st.cache_resource
def load_model():
    # Multilingual BERT (bert-base-multilingual-cased) fine-tuned for emotion detection
    return pipeline(
        "text-classification",
        model="AnasAlokla/multilingual_go_emotions",
        top_k=None,
    )

classifier = load_model()

examples = {
    "English": "I am so happy today, everything is going great!",
    "Spanish": "Estoy muy triste porque perdí mi trabajo.",
    "French": "Je suis tellement en colère contre cette décision.",
    "Arabic": "أنا ممتن جدا لمساعدتك",
}

choice = st.selectbox("Pick an example or write your own below:", ["(write my own)"] + list(examples))
default_text = examples.get(choice, "")
text = st.text_area("Enter a short sentence:", value=default_text, height=100)

if st.button("Detect emotion"):
    if not text.strip():
        st.warning("Please type a sentence first.")
    else:
        with st.spinner("BERT is reading your sentence..."):
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
