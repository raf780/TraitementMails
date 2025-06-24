# app.py
#!/usr/bin/env python3
"""
Streamlit app – Outlook .msg summariser with GPT-4.1-mini.

Launch:   streamlit run app.py

The OpenAI key lives in Streamlit Cloud secrets:
  OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
"""

import io
from datetime import datetime

import extract_msg
import openai
import pandas as pd
import streamlit as st


# ────────────────────────── GPT helper ──────────────────────────

def summarise_email(meta: dict, body: str) -> str:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    context = [
        f"Date: {meta['date']}",
        f"Objet: {meta['subject']}",
        f"Expéditeur: {meta['sender']}",
        f"Destinataire: {meta['to']}",
    ]
    if meta["attachments"]:
        context.append("Pièces jointes: " + ", ".join(meta["attachments"]))

    prompt = (
        "Résume le mail suivant en une seule ligne Excel.\n\n"
        + "\n".join(context)
        + "\n\nCorps:\n"
        + body
    )

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_tokens=2048,
        top_p=1,
    )

    return resp.choices[0].message.content.strip()


# ───────────────────────── .msg → record ─────────────────────────

def msg_bytes_to_record(raw: bytes, idx: int) -> dict:
    msg = extract_msg.Message(io.BytesIO(raw))

    meta = {
        "date": msg.date or "unknown",
        "subject": msg.subject or "sans objet",
        "sender": msg.sender or "inconnu",
        "to": msg.to or "inconnu",
        "attachments": [
            att.longFilename or att.shortFilename for att in msg.attachments
        ] if msg.attachments else [],
    }

    try:
        tag = datetime.strptime(meta["date"][:24], "%a %d %b %Y %H:%M:%S").strftime("%Y%m%d")
    except (ValueError, TypeError):
        tag = "unknown"
    numero = f"{tag}_{idx:03d}"

    synthese = summarise_email(meta, msg.body or "")

    return {
        "Numéro de l'email par date": numero,
        "Emetteur": meta["sender"],
        "Récipiendaire": meta["to"],
        "Synthèse": synthese,
    }


# ───────────────────────────── UI ───────────────────────────────

st.set_page_config(page_title="Synthèse emails Outlook", layout="wide")
st.title("Synthèse d'emails Outlook (.msg)")

files = st.file_uploader(
    "Glissez ici vos fichiers .msg (plusieurs possibles)",
    type="msg",
    accept_multiple_files=True,
)

if st.button("Lancer la synthèse") and files:
    rows = []
    for i, f in enumerate(files, 1):
        with st.spinner(f"Traitement {f.name}"):
            rows.append(msg_bytes_to_record(f.read(), i))

    df = pd.DataFrame(rows)
    st.success("Synthèse terminée")
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False)
    xlsx = buf.getvalue()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Télécharger CSV", csv, "synthese.csv", "text/csv")
    with col2:
        st.download_button(
            "Télécharger XLSX",
            xlsx,
            "synthese.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
