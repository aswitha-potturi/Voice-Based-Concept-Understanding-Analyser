import io
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


FILLER_WORDS = {
    "um",
    "uh",
    "erm",
    "ah",
    "like",
    "actually",
    "basically",
    "literally",
    "you know",
    "i mean",
}

REFERENCE_CONCEPTS = {
    "Machine Learning": (
        "Machine learning is a field of artificial intelligence where computer "
        "systems learn patterns from data and improve their performance on a task "
        "without being explicitly programmed for every rule. It includes supervised, "
        "unsupervised, and reinforcement learning approaches."
    ),
    "Cloud Computing": (
        "Cloud computing is the delivery of computing resources such as servers, "
        "storage, databases, networking, and software over the internet. It enables "
        "scalable, on-demand access, often through service models such as IaaS, "
        "PaaS, and SaaS."
    ),
    "Database Management System": (
        "A database management system is software used to define, store, retrieve, "
        "update, and manage structured data. It supports querying, integrity, "
        "security, concurrency control, and efficient organization of information."
    ),
    "Operating System": (
        "An operating system manages computer hardware and software resources. It "
        "provides services such as process management, memory management, file "
        "systems, device control, security, and a user interface for applications."
    ),
}


@dataclass
class AudioMetrics:
    duration: float
    rms_energy: float
    pause_ratio: float
    speech_ratio: float
    sample_rate: int
    waveform_image: bytes


def transcribe_audio(audio_path: str) -> tuple[str, str | None]:
    """Transcribe with Whisper when available; return a friendly warning otherwise."""
    try:
        import whisper
    except ImportError:
        return "", "Install openai-whisper to enable automatic speech transcription."

    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result.get("text", "").strip(), None
    except Exception as exc:
        return "", f"Whisper transcription failed: {exc}"


def semantic_similarity(user_text: str, reference_text: str) -> tuple[float, str]:
    """Use Sentence-BERT if installed, with a TF-IDF fallback for lightweight demos."""
    if not user_text.strip():
        return 0.0, "No transcript text was available for semantic comparison."

    try:
        from sentence_transformers import SentenceTransformer, util

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([user_text, reference_text], convert_to_tensor=True)
        score = float(util.cos_sim(embeddings[0], embeddings[1]).item())
        return max(0.0, min(score, 1.0)), "Sentence-BERT semantic embedding score"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([user_text, reference_text])
        score = float(cosine_similarity(matrix[0], matrix[1])[0][0])
        return max(0.0, min(score, 1.0)), "TF-IDF fallback similarity score"


def count_fillers(text: str) -> tuple[int, dict[str, int]]:
    lowered = f" {text.lower()} "
    counts: dict[str, int] = {}
    for filler in FILLER_WORDS:
        pattern = rf"(?<!\w){re.escape(filler)}(?!\w)"
        count = len(re.findall(pattern, lowered))
        if count:
            counts[filler] = count
    return sum(counts.values()), dict(sorted(counts.items()))


def analyze_audio(audio_path: str) -> AudioMetrics:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("Install librosa and soundfile to enable audio feature extraction.") from exc

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    rms = librosa.feature.rms(y=y)[0]
    rms_energy = float(np.mean(rms)) if len(rms) else 0.0

    intervals = librosa.effects.split(y, top_db=30)
    voiced_samples = sum(end - start for start, end in intervals)
    speech_ratio = float(voiced_samples / len(y)) if len(y) else 0.0
    pause_ratio = max(0.0, 1.0 - speech_ratio)

    fig, ax = plt.subplots(figsize=(10, 2.8))
    times = np.linspace(0, duration, num=len(y)) if len(y) else np.array([])
    ax.plot(times, y, color="#2563eb", linewidth=0.8)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format="png", dpi=160)
    plt.close(fig)

    return AudioMetrics(
        duration=duration,
        rms_energy=rms_energy,
        pause_ratio=pause_ratio,
        speech_ratio=speech_ratio,
        sample_rate=sr,
        waveform_image=image_buffer.getvalue(),
    )


def classify_understanding(score: float) -> str:
    if score >= 0.78:
        return "Strong Understanding"
    if score >= 0.52:
        return "Moderate Understanding"
    return "Poor Understanding"


def fluency_score(filler_count: int, pause_ratio: float, duration: float) -> float:
    words_penalty = min(filler_count * 2.5, 25)
    pause_penalty = min(pause_ratio * 45, 35)
    duration_penalty = 10 if duration < 10 else 0
    return max(0.0, 100.0 - words_penalty - pause_penalty - duration_penalty)


def final_score(similarity: float, fluency: float) -> float:
    return round((similarity * 100 * 0.72) + (fluency * 0.28), 2)


def generate_feedback(label: str, similarity: float, filler_count: int, pause_ratio: float) -> list[str]:
    feedback = []
    if label == "Strong Understanding":
        feedback.append("The explanation covers the concept with strong semantic alignment.")
    elif label == "Moderate Understanding":
        feedback.append("The explanation captures part of the concept but should include more core details.")
    else:
        feedback.append("The explanation appears to miss several essential ideas from the reference concept.")

    if similarity < 0.65:
        feedback.append("Add key terms, examples, and relationships that directly match the selected topic.")
    if filler_count > 5:
        feedback.append("Reduce filler words to make the explanation sound more confident and concise.")
    if pause_ratio > 0.45:
        feedback.append("Pause ratio is high; practice a smoother speaking rhythm with shorter hesitations.")
    if not feedback:
        feedback.append("The response is clear and well balanced.")
    return feedback


def create_pdf_report(
    topic: str,
    transcript: str,
    reference: str,
    similarity: float,
    similarity_method: str,
    metrics: AudioMetrics,
    filler_count: int,
    filler_breakdown: dict[str, int],
    fluency: float,
    comprehension_score: float,
    label: str,
    feedback: list[str],
) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
        from reportlab.lib import colors
    except ImportError as exc:
        raise RuntimeError("Install reportlab to enable downloadable PDF reports.") from exc

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Voice-Based Concept Understanding Analyser Report", styles["Title"]),
        Spacer(1, 0.18 * inch),
        Paragraph(f"<b>Topic:</b> {topic}", styles["Normal"]),
        Paragraph(f"<b>Classification:</b> {label}", styles["Normal"]),
        Spacer(1, 0.18 * inch),
    ]

    rows = [
        ["Metric", "Value"],
        ["Semantic similarity", f"{similarity:.2%} ({similarity_method})"],
        ["Fluency score", f"{fluency:.2f}/100"],
        ["Final comprehension score", f"{comprehension_score:.2f}/100"],
        ["Duration", f"{metrics.duration:.2f} seconds"],
        ["Pause ratio", f"{metrics.pause_ratio:.2%}"],
        ["RMS energy", f"{metrics.rms_energy:.5f}"],
        ["Filler words", f"{filler_count} {filler_breakdown}"],
    ]
    table = Table(rows, colWidths=[2.25 * inch, 4.4 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([table, Spacer(1, 0.22 * inch)])

    waveform_stream = io.BytesIO(metrics.waveform_image)
    story.extend([Paragraph("<b>Waveform Visualization</b>", styles["Heading2"]), Image(waveform_stream, width=6.6 * inch, height=1.85 * inch)])
    story.extend([Spacer(1, 0.16 * inch), Paragraph("<b>Transcript</b>", styles["Heading2"]), Paragraph(transcript or "No transcript available.", styles["BodyText"])])
    story.extend([Spacer(1, 0.16 * inch), Paragraph("<b>Reference Concept</b>", styles["Heading2"]), Paragraph(reference, styles["BodyText"])])
    story.extend([Spacer(1, 0.16 * inch), Paragraph("<b>Feedback</b>", styles["Heading2"])])
    for item in feedback:
        story.append(Paragraph(f"- {item}", styles["BodyText"]))

    doc.build(story)
    return buffer.getvalue()


def write_temp_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def main() -> None:
    st.set_page_config(page_title="VBCUA", page_icon="V", layout="wide")
    st.title("Voice-Based Concept Understanding Analyser (VBCUA)")
    st.caption("Evaluate conceptual understanding and speech fluency from spoken explanations.")

    with st.sidebar:
        st.header("Evaluation Setup")
        topic = st.selectbox("Reference concept", list(REFERENCE_CONCEPTS))
        custom_reference = st.text_area("Custom reference concept", value=REFERENCE_CONCEPTS[topic], height=170)
        uploaded_audio = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg", "flac"])
        manual_transcript = st.text_area(
            "Optional transcript override",
            help="Use this when Whisper is not installed or when you want to correct the transcript.",
            height=130,
        )
        run_analysis = st.button("Run Evaluation", type="primary", use_container_width=True)

    st.subheader("Audio Input")
    if uploaded_audio:
        st.audio(uploaded_audio)
    else:
        st.info("Upload a spoken explanation to begin.")

    if not run_analysis:
        st.stop()

    if not uploaded_audio:
        st.error("Please upload an audio file before running evaluation.")
        st.stop()

    audio_path = write_temp_upload(uploaded_audio)
    reference_text = custom_reference.strip() or REFERENCE_CONCEPTS[topic]

    with st.spinner("Transcribing and analysing speech..."):
        auto_transcript, transcription_warning = transcribe_audio(audio_path)
        transcript = manual_transcript.strip() or auto_transcript
        similarity, similarity_method = semantic_similarity(transcript, reference_text)
        filler_count, filler_breakdown = count_fillers(transcript)
        metrics = analyze_audio(audio_path)
        fluency = fluency_score(filler_count, metrics.pause_ratio, metrics.duration)
        comprehension_score = final_score(similarity, fluency)
        label = classify_understanding(similarity)
        feedback = generate_feedback(label, similarity, filler_count, metrics.pause_ratio)

    if transcription_warning and not manual_transcript.strip():
        st.warning(transcription_warning)

    metric_cols = st.columns(5)
    metric_cols[0].metric("Semantic Similarity", f"{similarity:.1%}")
    metric_cols[1].metric("Fluency Score", f"{fluency:.1f}/100")
    metric_cols[2].metric("Final Score", f"{comprehension_score:.1f}/100")
    metric_cols[3].metric("Pause Ratio", f"{metrics.pause_ratio:.1%}")
    metric_cols[4].metric("Filler Words", str(filler_count))

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Transcript")
        st.write(transcript or "No transcript available. Add a transcript override or install Whisper.")
        st.subheader("Waveform Visualization")
        st.image(metrics.waveform_image, use_container_width=True)

    with right:
        st.subheader("Evaluation")
        st.success(label)
        st.write(f"Similarity method: {similarity_method}")
        st.write(f"Duration: {metrics.duration:.2f}s")
        st.write(f"Sample rate: {metrics.sample_rate} Hz")
        st.write(f"RMS energy: {metrics.rms_energy:.5f}")
        st.write(f"Filler breakdown: {filler_breakdown or 'None detected'}")
        st.subheader("Structured Feedback")
        for item in feedback:
            st.write(f"- {item}")

    st.subheader("Reference Concept")
    st.write(reference_text)

    try:
        pdf_bytes = create_pdf_report(
            topic,
            transcript,
            reference_text,
            similarity,
            similarity_method,
            metrics,
            filler_count,
            filler_breakdown,
            fluency,
            comprehension_score,
            label,
            feedback,
        )
        st.download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name=f"vbcua_{topic.lower().replace(' ', '_')}_report.pdf",
            mime="application/pdf",
        )
    except RuntimeError as exc:
        st.warning(str(exc))


if __name__ == "__main__":
    main()
