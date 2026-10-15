# 🇵🇭 Kuwentuhan Taglish AI Tutor 🇬🇧

[![Live Demo](https://img.shields.io/badge/Render-Live%20Demo-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://chatbot-kwentuhan-english-tagalog.onrender.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

An interactive bilingual conversation coach and language tutor specializing in **Tagalog (Filipino)**, **English**, and **everyday Taglish**. Built with FastAPI, streaming Server-Sent Events (SSE), and a minimalist dark charcoal & silver design.

🌐 **Live Demo:** [https://chatbot-kwentuhan-english-tagalog.onrender.com](https://chatbot-kwentuhan-english-tagalog.onrender.com)

---

## ✨ Features

- **Bilingual Learning Modes:**
  - 🇵🇭 **Filipino (Tagalog):** Practice Tagalog grammar, affixes (*mag-*, *nag-*, *-um-*), and polite particles (*po/opo*).
  - 🇬🇧 **English:** Polish everyday grammar, pronunciation, and correct common Filipinisms.
  - 💬 **Taglish:** Master natural Filipino-English code-switching used in daily conversations.
- **Strict Language Restrictions:** Dedicated solely to English and Tagalog. Foreign language questions are politely redirected.
- **Interactive Grammar Comparison Cards:** Markdown tables automatically render into boxed cards with distinct columns.
- **Auto-Resizing Message Box:** Input expands dynamically with multi-line support (`Shift + Enter`).
- **Interactive Suggestions Carousel:** Horizontal swipeable/scrollable carousel with starter topics.
- **Admin & Analytics Dashboard (`/admin`):** Conversation logs, ratings, and CSV export.

---

## 🚀 Quick Start (Local Development)

### 1. Clone the repository
```bash
git clone https://github.com/JohnLester0/ChatBot_kwentuhan_English_-_Tagalog.git
cd ChatBot_kwentuhan_English_-_Tagalog
```

### 2. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp env.example .env
```
Edit `.env` to configure your LLM provider (`openai` or `anthropic`) and API key.

### 4. Run the application
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
Open your browser at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## 🌐 Deployment

This application is deployed on [Render](https://render.com) as a Web Service:
* **Runtime:** Python 3 (or Docker)
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
* **Health Check Endpoint:** `/healthz`

---

## 📄 License
MIT License
