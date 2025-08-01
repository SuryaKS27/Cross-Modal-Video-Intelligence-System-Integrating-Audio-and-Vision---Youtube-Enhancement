## Description
This project is a cross-modal video intelligence system that enables voice-based semantic search and cognitive content analysis for YouTube videos. It integrates advanced AI models to analyze and enhance video content by transcribing speech (Whisper), detecting objects (YOLOv9), ranking results using semantic similarity (BERT), and improving video clarity (Real-ESRGAN). The system supports synchronized object and speech detection, entity extraction, summarization, and cognitive graph visualization for deeper content understanding.

When tested on 30 curated queries, the BERT-based semantic ranking achieved a 74% user preference over YouTube’s native search results, demonstrating improved relevance and usability. Built with Python, HTML/CSS, and the YouTube API, this solution is ideal for education, research, and media analysis where quick, meaningful insights from video content are essential.

## Table of Contents
[Installation](#installation)

[Deployment](#deployment)

[Tools](#tools)

[Results](#results)

[Resources](#resources)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

### 2. Create and Activate Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` - for installing manually:
```bash
pip install flask youtube_dl yt_dlp openai-whisper spacy transformers torch torchvision torchaudio ultralytics real-esrgan networkx matplotlib ffmpeg-python
```

### 4. Download spaCy Model
```bash
python -m spacy download en_core_web_sm
```

### 5. Install Real-ESRGAN (Video Enhancement)
```bash
pip install realesrgan
```

### 6. Install FFmpeg (Required for Video Processing)
Download FFmpeg from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) (Windows build).  

## Deployment

### 1. Set Environment Variables
Set your YouTube API key in **app.py**
```powershell
YOUTUBE_API_KEY = "your_api_key_here"
```

### 2. Run the Application
```bash
python app.py
```

### 3. Access the Web Interface
Open your browser and go to:
```
http://127.0.0.1:5000/

```
## Tools

- **Programming Language:** Python
- **Frontend:** HTML, CSS
- **APIs:** YouTube Data API v3
- **Speech Recognition:** OpenAI Whisper
- **Natural Language Processing:** spaCy, BERT, BART
- **Object Detection:** YOLOv9
- **Video Enhancement:** Real-ESRGAN
- **Data Visualization:** NetworkX, Matplotlib
- **Video Processing:** FFmpeg

## Results

- Achieved **74% user preference** over YouTube’s native search results.
- Evaluated on **30 curated queries** for semantic relevance.
- Improved video clarity and content understanding with integrated AI models.
- Enabled synchronized object and speech detection for enhanced context analysis.
- Generated concise and accurate video summaries using BART.
- Produced cognitive graph visualizations for deeper semantic insights.

### Model Training & Evaluation
- Developed a **GAN-based abstractive summarization framework** using BART (Generator) and BERT (Discriminator).
- Trained on a reduced **CNN/DailyMail dataset** with 5,000 samples for 2 epochs, batch size 8, and input sequence length of 512 tokens.

**Training Dynamics:**
- **Epoch 0:** Generator Loss reduced from `11.8762` to `4.3228`, while Discriminator Loss varied between `0.7189` and `0.5668`, indicating improved Generator outputs and balanced adversarial learning.
- **Epoch 1:** Generator Loss further dropped to `1.2895`–`2.0946`, with Discriminator Loss ranging from `0.0306` to `0.4727`, showing the Generator producing highly realistic and semantically accurate summaries.
- Steady decline in Generator loss confirmed progressive improvement in summary generation.
- Gradual increase in Discriminator difficulty indicated Generator’s outputs were becoming harder to distinguish from real summaries.

**Qualitative Evaluation:**
- Summarized a sample news article about the Amazon rainforest with a concise and fluent output.
- Retained core meaning while removing unnecessary numerical and causal details.

**Observations:**
- Adversarial training significantly improved summary quality.
- No signs of instability or mode collapse.
- Generated summaries became more human-like as training progressed.

## Resources

- **Project Report:** [View Report](https://github.com/SuryaKS27/Cross-Modal-Video-Intelligence-System-Integrating-Audio-and-Vision---Youtube-Enhancement/blob/1362ff068c7f355e8fd6ea664db37d5e012b3695/Project%20Report.pdf)
- **Project Demo:** [Watch Demo](https://drive.google.com/file/d/1nP43fF-41fwQnVr5UtX_QmWKA9emL---/view?pli=1)

## Author

[**Surya K S**]([https://github.com/SuryaKS27])  


  


