#!/usr/bin/env python3
"""
app.py – Résumé d’e-mails Outlook (.msg) avec GPT-4.1-mini
Déploiement Streamlit Cloud : ajoutez le secret
    OPENAI_API_KEY = "sk-…"
dans l’onglet *Secrets* de l’app.
"""

import io
from datetime import datetime

import extract_msg
import openai
import pandas as pd
import streamlit as st
from dateutil import parser as dtparser


# ──────────────────────── GPT helper ────────────────────────
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
        temperature=0.8,
        max_tokens=2048,
        top_p=1,
    )

    return resp.choices[0].message.content.strip()


# ─────────────── .msg → enregistrement DataFrame ─────────────
def msg_bytes_to_record(raw: bytes, idx: int, fname: str) -> dict:
    msg = extract_msg.Message(io.BytesIO(raw))

    meta = {
        "date": msg.date or "",
        "subject": msg.subject or "sans objet",
        "sender": msg.sender or "inconnu",
        "to": msg.to or "inconnu",
        "attachments": [
            att.longFilename or att.shortFilename for att in msg.attachments
        ] if msg.attachments else [],
    }

    # essai de lecture de la date du mail
    try:
        date_tag = dtparser.parse(meta["date"], fuzzy=True).strftime("%Y%m%d")
    except Exception:
        date_tag = datetime.now().strftime("%Y%m%d")

    numero = f"{date_tag}_{idx:03d}"
    synthese = summarise_email(meta, msg.body or "")

    return {
        "Numéro de l'email par date": numero,
        "Emetteur": meta["sender"],
        "Récipiendaire": meta["to"],
        "Synthèse": synthese,
    }


# ─────────────────────────── UI ────────────────────────────
st.set_page_config(page_title="Synthèse emails Outlook", layout="wide")
st.title("Synthèse d'emails Outlook (.msg)")

files = st.file_uploader(
    "Glissez ici vos fichiers .msg",
    type="msg",
    accept_multiple_files=True,
)

if st.button("Lancer la synthèse") and files:
    rows = []
    for i, f in enumerate(files, 1):
        with st.spinner(f"Traitement {f.name}"):
            rows.append(msg_bytes_to_record(f.read(), i, f.name))

    df = pd.DataFrame(rows)
    st.success("Synthèse terminée")
    st.dataframe(df, use_container_width=True)

    csv_data = df.to_csv(index=False).encode("utf-8")
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False)
    xlsx_data = xlsx_buf.getvalue()

    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Télécharger CSV", csv_data, "synthese.csv", "text/csv")
    with col2:
        st.download_button(
            "Télécharger XLSX",
            xlsx_data,
            "synthese.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
