import os
import re
import tempfile
import streamlit as st
from transformers import pipeline


# -----------------------------
# Utils
# -----------------------------
def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(story:|answer:)\s*", "", text, flags=re.IGNORECASE)
    return text


def has_good_ending(text: str) -> bool:
    t = text.lower().strip()
    ending_keywords = [
        "kind", "kindness", "share", "sharing", "brave", "courage",
        "friend", "friends", "lesson", "learned", "learn"
    ]
    return any(k in t for k in ending_keywords) and t.endswith((".", "!", "?"))


def fallback_story(caption: str) -> str:
    return (
        f"One sunny afternoon, the children saw {caption} and ran to the park with big smiles. "
        "They played a ball game, took turns, and cheered for one another. "
        "When a new child stood alone, they invited him to join and showed him the rules. "
        "Soon everyone was laughing and helping each other. "
        "As the sky turned orange, they walked home happily. "
        "They learned that kindness and sharing make every game more fun."
    )


# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_pipelines():
    img_pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    story_pipe = pipeline("text2text-generation", model="google/flan-t5-base")
    tts_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    return img_pipe, story_pipe, tts_pipe


# -----------------------------
# Core functions
# -----------------------------
def img2text(image_path: str) -> str:
    img_pipe, _, _ = load_pipelines()
    result = img_pipe(image_path)
    return result[0]["generated_text"]


def generate_story_once(caption: str) -> str:
    _, story_pipe, _ = load_pipelines()

    prompt = (
        "Write one complete children's story for ages 3-10.\n"
        "Rules:\n"
        "1) 50 to 100 words.\n"
        "2) Use short, simple, clear sentences.\n"
        "3) Warm and positive tone.\n"
        "4) No violence, no horror, no scary content.\n"
        "5) Include a clear beginning, middle, and ending.\n"
        "6) Last sentence gives a gentle lesson about kindness, sharing, or courage.\n\n"
        f"Image description: {caption}\n"
        "Story:"
    )

    out = story_pipe(
        prompt,
        max_new_tokens=180,
        do_sample=True,
        temperature=0.85,
        top_p=0.9
    )[0]["generated_text"]

    return clean_text(out)


def text2story(caption: str, max_attempts: int = 5) -> str:
    best_story = ""
    best_score = -1

    for _ in range(max_attempts):
        story = generate_story_once(caption)
        wc = word_count(story)

        score = 0
        if 50 <= wc <= 100:
            score += 2
        if has_good_ending(story):
            score += 2
        sentence_count = len([s for s in re.split(r"[.!?]+", story) if s.strip()])
        if sentence_count >= 4:
            score += 1

        if score > best_score:
            best_score = score
            best_story = story

        if score >= 4:
            return story

    if not best_story or word_count(best_story) < 50 or not has_good_ending(best_story):
        return fallback_story(caption)

    words = best_story.split()
    if len(words) > 100:
        best_story = " ".join(words[:100]).rstrip(",;:") + "."

    if word_count(best_story) < 50:
        return fallback_story(caption)

    return best_story


def text2audio(story_text: str):
    _, _, tts_pipe = load_pipelines()
    return tts_pipe(story_text)


# -----------------------------
# Streamlit app
# -----------------------------
def main():
    st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜", layout="centered")
    st.title("Turn Your Image to Audio Story")
    st.write("This app generates a child-friendly story (ages 3–10) and audio from an image.")

    st.write("## Step 1: Choose image input")
    mode = st.radio(
        "Input method:",
        ["Upload image", "Use filename in current working directory"],
        horizontal=True
    )

    image_path = None
    temp_file_path = None

    if mode == "Upload image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded_file is not None:
            suffix = os.path.splitext(uploaded_file.name)[1] or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                temp_file_path = tmp.name
                image_path = temp_file_path

            st.image(uploaded_file, caption="Uploaded image", use_container_width=True)
            st.write("Image ready.")
    else:
        filename = st.text_input("Enter image filename:", value="test.jpg")
        if filename:
            if os.path.exists(filename):
                image_path = filename
                st.image(filename, caption=f"Loaded: {filename}", use_container_width=True)
                st.write("Image ready.")
            else:
                st.write(f"File not found: {filename}")

    if st.button("Generate Story & Audio", type="primary"):
        if not image_path:
            st.write("Please provide an image first.")
            return

        try:
            st.write("Processing img2text...")
            caption = img2text(image_path)
            st.write(caption)

            st.write("Generating a story...")
            story = text2story(caption)
            st.write(story)
            st.write(f"Word count: {word_count(story)}")

            st.write("Generating audio data...")
            audio_data = text2audio(story)

            st.audio(audio_data["audio"], sample_rate=audio_data["sampling_rate"])
            st.write("Done.")

            st.download_button(
                "Download story text",
                data=story,
                file_name="story.txt",
                mime="text/plain"
            )

        except Exception as e:
            st.write(f"Error: {e}")

        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
