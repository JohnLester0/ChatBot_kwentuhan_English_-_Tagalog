# ChatBot_kwentuhan_English_-_Tagalog

# Kuwentuhan Bot
Filipino / English / Taglish chatbot with jokes and stories. Runs on an LLM API (or local Ollama).

## Run
    pip install -r requirements.txt
    uvicorn app:app --reload    # http://127.0.0.1:8000

## Pages
- `/`       chat app (language modes, jokes, stories, thumbs up/down)
- `/admin`  stats + CSV export (needs ADMIN_TOKEN)
- `/health` status check

## Notes
- Chats are saved to SQLite for research when LOG_CHATS=true. The chat page tells users this.
- Free hosts often wipe disk on restart: export the CSV regularly or attach a persistent disk.
- Rate limiting is in-memory, so run a single worker.
