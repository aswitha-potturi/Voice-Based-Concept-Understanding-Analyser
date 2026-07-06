<div align="center">

# 🎙️ VBCUA — Voice-Based Concept Understanding Analyser

**Evaluate how well someone understands a concept — just by listening to them explain it.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)

</div>

---

## 📖 Overview

**VBCUA** is a Streamlit application that turns a spoken explanation into a structured comprehension report. Upload an audio recording of someone explaining a concept, and VBCUA transcribes it, compares it semantically to a reference explanation, analyzes speech fluency, and generates a downloadable PDF report — complete with waveform visualization and actionable feedback.

Perfect for **educators, trainers, interviewers, and self-learners** who want an objective, repeatable way to assess spoken understanding.

---

## ✨ Features

| Category | Capability |
|---|---|
| 🎧 **Audio Input** | Upload and play back spoken explanations directly in-app |
| 📝 **Transcription** | Automatic speech-to-text via OpenAI Whisper |
| 🧠 **Semantic Scoring** | Sentence-BERT similarity against a reference concept |
| 🔁 **Graceful Fallback** | Automatic TF-IDF similarity if Sentence-BERT isn't available |
| 🗣️ **Fluency Analysis** | Detects filler words (`um`, `uh`, `like`, `you know`, etc.) |
| 📊 **Audio Metrics** | Pause ratio, RMS energy, and more via Librosa |
| 🌊 **Waveform Visualization** | Interactive waveform rendered inside the dashboard |
| 📄 **PDF Reports** | One-click downloadable evaluation report via ReportLab |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- (Optional) A GPU speeds up Whisper transcription, but isn't required

### Installation

```powershell
# 1. Clone the repository
git clone https://github.com/<your-username>/vbcua.git
cd vbcua

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

> 💡 **Tip — Lightweight demo mode:** Whisper and Sentence-BERT models can take a while to download on first run. If you want a fast, minimal setup, skip `openai-whisper`, `torch`, and `sentence-transformers` in `requirements.txt`, and simply paste a transcript into the transcript override box instead of uploading audio.

---

## 🧭 How It Works

```
   ① Select Concept   →   ② Upload Audio   →   ③ Transcribe
          ↓                                          ↓
   ⑥ Download PDF   ←   ⑤ Review Feedback   ←   ④ Run Evaluation
```

1. **Select a reference concept** or write/edit your own reference text.
2. **Upload a spoken explanation** as an audio file.
3. **Optionally paste or correct** the auto-generated transcript.
4. **Run the evaluation.**
5. **Review results** — semantic score, fluency score, waveform, filler-word stats, pause analysis, and structured feedback.
6. **Download the PDF report** to save or share.

---

## 🧮 Scoring Methodology

The final **Comprehension Score** is a weighted blend:

| Component | Weight | Based On |
|---|---|---|
| 🧠 Semantic Similarity | **72%** | Closeness to the reference concept |
| 🗣️ Fluency Score | **28%** | Filler words, pause ratio, and recording length |

### Understanding Labels

| Label | Semantic Similarity Range |
|---|---|
| 🟢 **Strong Understanding** | 78% and above |
| 🟡 **Moderate Understanding** | 52% – 77% |
| 🔴 **Poor Understanding** | Below 52% |

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** — interactive web dashboard
- **[OpenAI Whisper](https://github.com/openai/whisper)** — speech-to-text transcription
- **[Sentence-BERT](https://www.sbert.net/)** — semantic similarity embeddings
- **[Librosa](https://librosa.org/)** — audio feature extraction
- **[ReportLab](https://www.reportlab.com/)** — PDF report generation

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

<div align="center">

Made with 🎙️ for better spoken understanding.

</div>
