import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
from huggingface_hub import InferenceClient

st.set_page_config(page_title="inapinep.id", layout="centered")

st.markdown("""<style>
@import url("https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600&display=swap");
html, body, .stApp { background-color: #fff0f5 !important; color: #4a1942 !important; font-family: Nunito, sans-serif !important; }
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stChatInput"] textarea { background: #fff5f8 !important; color: #4a1942 !important; border: 2px solid #e8a0b4 !important; border-radius: 20px !important; }
.stButton button { background: linear-gradient(135deg, #e8a0b4, #d4608a) !important; color: white !important; border: none !important; border-radius: 20px !important; font-weight: 700 !important; }
[data-testid="stChatMessage"] { background: #fff5f8 !important; border: 1px solid #f0c4d4 !important; border-radius: 20px !important; padding: 14px !important; margin: 6px 0 !important; }
</style>""", unsafe_allow_html=True)

st.markdown("""<div style="text-align:center;padding:2rem 0 1rem;">
<div style="font-family:serif;font-size:2.4rem;font-weight:700;background:linear-gradient(135deg,#e8a0b4,#d4608a);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">inapinep.id</div>
<div style="color:#b06080;font-size:0.9rem;margin-top:8px;">Temukan hotel impianmu ✨ · Pulau Jawa 🌷 · Rekomendasi fasilitas 💆‍♀️ · Info harga & rating 💕</div>
<div style="margin-top:12px;display:inline-block;background:#fff5f8;border:1px solid #e8a0b4;padding:4px 16px;border-radius:20px;font-size:0.75rem;color:#d4608a;"> ONLINE</div>
</div>""", unsafe_allow_html=True)

@st.cache_resource
def load_all_data():
    base = "/content/drive/MyDrive/hotel-chatbot"
    data_dir = os.path.join(base, "data")
    df = pd.read_csv(os.path.join(data_dir, "dataset_hotel_indonesia.csv"))
    df = df.fillna("")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["facilities"] = df["facilities"].apply(lambda x: re.sub(r"[^a-zA-Z0-9 ,]", "", str(x)).strip())
    df["text"] = (df["hotel_name"] + " " + df["property_type"] + " " + df["city"] + " " + df["facilities"]).str.lower()
    return df

try:
    df = load_all_data()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

HF_TOKEN = os.environ.get("HF_TOKEN", "")

def ask_hf(messages):
    models_to_try = ["mistralai/Mixtral-8x7B-Instruct-v0.1","Qwen/Qwen2.5-72B-Instruct","google/gemma-2-2b-it","HuggingFaceH4/zephyr-7b-beta"]
    client = InferenceClient(token=HF_TOKEN)
    for model in models_to_try:
        try:
            response = client.chat_completion(model=model, messages=messages, max_tokens=700, temperature=0.7)
            return response.choices[0].message.content
        except Exception:
            continue
    return "Aduh maaf, chatbot sedang tidak tersedia. Coba lagi beberapa menit yaa! 🌸"

def cari_hotel(query, kota=None, max_harga=None, min_rating=None):
    hasil = df.copy()
    if kota: hasil = hasil[hasil["city"].str.contains(kota, case=False, na=False)]
    if max_harga: hasil = hasil[hasil["min_price"] <= max_harga]
    if min_rating: hasil = hasil[hasil["rating"] >= min_rating]
    if query: hasil = hasil[hasil["text"].str.contains(query.lower(), na=False)]
    return hasil.sort_values("rating", ascending=False).head(5)

def format_hotel(row):
    return (
        "<div style='background:#fff5f8;border:1px solid #f0c4d4;border-radius:16px;padding:14px;margin:8px 0;box-shadow:0 2px 8px rgba(232,160,180,0.1);'>" +
        f"<div style='font-size:1rem;font-weight:700;color:#d4608a;'>🏩 {row['hotel_name']}</div>" +
        f"<div style='margin-top:6px;font-size:0.85rem;color:#7a4060;'>📍 {row['city']}<br>⭐ Rating: <b>{row['rating']}</b><br>💰 Harga: Rp {int(row['min_price']):,} - Rp {int(row['max_price']):,}<br>🏷️ Tipe: {row['property_type']}</div>" +
        "</div>"
    )

SYSTEM_PROMPT = """inapinep.id  — asisten rekomendasi hotel Indonesia yang cerdas, ramah, dan friendly untuk perempuan.
Kamu membantu pengguna menemukan hotel terbaik di Indonesia berdasarkan kota, fasilitas, budget, dan rating.
Kamu bisa rekomendasikan hotel di kota tertentu, cari berdasarkan fasilitas, info budget, dan tips memilih hotel.
Format jawaban: Bahasa Indonesia yang hangat, ramah, dan encouraging! Gunakan emoji yang sesuai 🌸✨💕"""

if "messages" not in st.session_state: st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.markdown("""<div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:1.5rem;">
    <div style="background:#fff5f8;border:1px solid #e8a0b4;padding:8px 16px;border-radius:20px;font-size:0.82rem;color:#d4608a;">Hotel di Surabaya yang romantis</div>
    <div style="background:#fff5f8;border:1px solid #d4608a;padding:8px 16px;border-radius:20px;font-size:0.82rem;color:#b04070;">Hotel murah di Jakarta</div>
    <div style="background:#fff5f8;border:1px solid #f0a0c0;padding:8px 16px;border-radius:20px;font-size:0.82rem;color:#c05080;">Hotel dengan spa & kolam renang</div>
    <div style="background:#fff5f8;border:1px solid #c080a0;padding:8px 16px;border-radius:20px;font-size:0.82rem;color:#905070;">Tips memilih hotel yang nyaman</div>
    </div>""", unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

if prompt := st.chat_input("Penginapan terbaik khusus untukmu 🌸"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Mencarikan hotel terbaik untukmu 🌸..."):
            kota_keywords = ["surabaya","jakarta","semarang","bandung","yogyakarta","malang","madiun","gresik","pacitan","magelang"]
            kota_found = next((k for k in kota_keywords if k in prompt.lower()), None)
            hotel_results = cari_hotel(prompt, kota=kota_found)
            context = ""
            if len(hotel_results) > 0:
                hotel_cards = "".join([format_hotel(row) for _, row in hotel_results.iterrows()])
                context = f"\nData hotel relevan: {hotel_results[['hotel_name','city','rating','min_price','max_price','property_type']].to_string()}"
                st.markdown(hotel_cards, unsafe_allow_html=True)
            ai_messages = [{"role": "system", "content": SYSTEM_PROMPT + context}]
            for m in st.session_state.messages:
                ai_messages.append({"role": m["role"], "content": m["content"]})
            full_reply = ask_hf(ai_messages)
            st.markdown(full_reply, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": full_reply})

if st.session_state.messages:
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.rerun()
