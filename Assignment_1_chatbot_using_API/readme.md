# Assignment 1: Chatbot Using API

## Overview
This project demonstrates how to build and enhance a chatbot through **different API integration methods**, exploring multiple versions and functionalities.  
We started from a simple Gemini API chatbot and gradually developed it into a **universal multi-provider chatbot** supporting various AI backends.

---

## Different Phases

### **Version 01 — Terminal-Based Gemini Chatbot**
- Used **Gemini API** with model **`gemini-2.5-flash`**.  
- User interacts via **terminal**, asks a question, and receives responses directly.  
- Typing **`exit`** or **`quit`** ends the session.  

### **Version 02 — Streamlit (Single-Turn Chatbot)**
- Implemented chatbot using **Streamlit** for a web interface.  
- The app asks for the **API key** directly from the user for privacy.  
- Responds to **one query at a time** (no memory of previous messages).

### **Version 03 — Continuous Chat Interface**
- Added Streamlit’s **`st.chat_message()`** for a chat-style UI.  
- Chat messages display visually like real conversations.  
- However, the model still doesn’t access previous context—each input is treated as new.

### **Version 04 — Chat Memory (Context-Aware Chatbot)**
- Introduced **chat history memory** so the chatbot can remember previous messages.  
- Generates **context-aware** responses based on full conversation history.  
- Provides a more natural, human-like chat experience.

### **Version 05 & 06 — Universal Multi-API Chatbot**
- Enhanced chatbot to support **multiple AI providers** for greater flexibility:
  - Gemini  
  - OpenAI  
  - Groq  
  - Hugging Face  
  - Ollama / Custom APIs  
- Users can:
  - Select any **API provider**
  - Specify **model name** (e.g., `gpt-4`, `llama3-7b`, etc.)
  - Enter their **API key**
- Implemented **dynamic imports (`importlib`)** and **REST API calls** for universal compatibility.

---

## Feature Comparison

| Feature        | Old Version            | Latest Version |
|----------------|------------------------|----------------|
| **Provider**   | Only Gemini             | Multiple (Gemini, OpenAI, Groq, HF, etc.) |
| **API Handling** | Static                 | Dynamic with `importlib` + REST |
| **Conversation** | Context-unaware        | Full memory retained |
| **UI**         | Simple Streamlit input  | Chat-style interface with bubbles |
| **Extensibility** | Fixed                 | Easily add new providers |

