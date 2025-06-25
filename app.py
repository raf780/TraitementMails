#!/usr/bin/env python3
"""
app.py – Résumé d’e-mails .msg **et** .eml avec GPT-4.1-mini (Streamlit).

Déploiement Streamlit Cloud :
  1. Poussez ce fichier + requirements.txt dans votre repo.
  2. Dans l’onglet *Secrets*, ajoutez :
       OPENAI_API_KEY = "sk-…"
"""

import io
from datetime import datetime
from email import policy
from email.parser import BytesParser
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        # Ensure all attachment names are strings and filter out None values
        safe_attachments = [str(att) for att in meta["attachments"] if att is not None]
        if safe_attachments:
            context.append("Pièces jointes: " + ", ".join(safe_attachments))

    prompt = (
        "Résume le mail suivant en une seule ligne explicative.\n\n"
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


# ──────────────── Helpers de parsing ────────────────
def parse_msg(raw: bytes):
    try:
        m = extract_msg.Message(io.BytesIO(raw))
    except Exception as e:
        raise Exception(f"Impossible de lire le fichier .msg: {str(e)}")

    # Extract attachment names, filtering out None values
    attachments = []
    if m.attachments:
        for att in m.attachments:
            filename = att.longFilename or att.shortFilename
            if filename:
                attachments.append(filename)
            else:
                attachments.append("fichier_sans_nom")

    meta = {
        "date": m.date,  # Garder None si pas de date, plutôt qu'une chaîne vide
        "subject": m.subject or "sans objet",
        "sender": m.sender or "inconnu",
        "to": m.to or "inconnu",
        "attachments": attachments,
    }
    body = m.body or ""
    return meta, body


def parse_eml(raw: bytes):
    eml = BytesParser(policy=policy.default).parsebytes(raw)

    meta = {
        "date": eml.get("date"),  # Garder None si pas de date, plutôt qu'une chaîne vide
        "subject": eml.get("subject", "sans objet"),
        "sender": eml.get("from", "inconnu"),
        "to": eml.get("to", "inconnu"),
        "attachments": [
            part.get_filename() or "fichier_sans_nom"
            for part in eml.iter_attachments()
            if part.get_filename()
        ],
    }

    # cherche la première partie text/plain non jointe
    body = ""
    if eml.is_multipart():
        for part in eml.walk():
            if (
                part.get_content_type() == "text/plain"
                and part.get_filename() is None
            ):
                body = part.get_content()
                break
    else:
        body = eml.get_content()

    return meta, body


# ─────────────── .msg/.eml → enregistrement DataFrame ─────────────
def file_bytes_to_record(raw: bytes, idx: int, fname: str) -> dict:
    ext = fname.lower().rsplit(".", 1)[-1]

    try:
        if ext == "msg":
            meta, body = parse_msg(raw)
        elif ext == "eml":
            meta, body = parse_eml(raw)
        else:
            raise ValueError("Extension non prise en charge")
    except Exception as e:
        # Si le fichier ne peut pas être parsé, retourner un enregistrement d'erreur
        st.error(f"Erreur lors du traitement de {fname}: {str(e)}")
        date_tag = datetime.now().strftime("%Y%m%d")
        numero = f"{date_tag}_{idx:03d}"
        return {
            "Numéro de l'email par date": numero,
            "Emetteur": "ERREUR",
            "Récipiendaire": "ERREUR",
            "Objet": "ERREUR",
            "Synthèse": f"Erreur lors du traitement du fichier {fname}: {str(e)}",
        }

    # format AAAAMMJJ - utiliser la date de l'email
    date_tag = None
    
    # Essayer de parser la date de l'email
    if meta["date"]:
        try:
            # Si c'est déjà un objet datetime
            if isinstance(meta["date"], datetime):
                date_tag = meta["date"].strftime("%Y%m%d")
            # Si c'est une chaîne, la parser
            elif isinstance(meta["date"], str) and meta["date"].strip():
                parsed_date = dtparser.parse(meta["date"], fuzzy=True)
                date_tag = parsed_date.strftime("%Y%m%d")
        except Exception as e:
            st.warning(f"Impossible de parser la date '{meta['date']}' pour {fname}, utilisation de 'date_inconnue'")
    
    # Fallback si aucune date n'est disponible
    if not date_tag:
        date_tag = "date_inconnue"
        st.warning(f"Aucune date trouvée pour {fname}, utilisation de 'date_inconnue'")

    numero = f"{date_tag}_{idx:03d}"
    
    try:
        synthese = summarise_email(meta, body)
    except Exception as e:
        st.error(f"Erreur lors de la synthèse de {fname}: {str(e)}")
        synthese = f"Erreur lors de la synthèse: {str(e)}"

    return {
        "Numéro de l'email par date": numero,
        "Emetteur": meta["sender"],
        "Récipiendaire": meta["to"],
        "Objet": meta["subject"],
        "Synthèse": synthese,
    }


# ─────────────────────────── UI ────────────────────────────
st.set_page_config(page_title="Synthèse d'emails", layout="wide")
st.title("Synthèse d'emails")

files = st.file_uploader(
    "Déposez vos fichiers .msg ou .eml",
    type=("msg", "eml"),
    accept_multiple_files=True,
)

if st.button("Lancer la synthèse") and files:
    # Configuration pour la parallélisation
    max_workers = st.sidebar.slider("Nombre de traitements parallèles", 1, 10, 3, 
                                   help="Plus élevé = plus rapide, mais attention aux limites de l'API OpenAI")
    
    # Préparer les données pour le traitement parallèle
    file_data = [(f.read(), i, f.name) for i, f in enumerate(files, 1)]
    
    # Traitement parallèle
    rows = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre tous les jobs
        future_to_index = {
            executor.submit(file_bytes_to_record, data[0], data[1], data[2]): data[1] 
            for data in file_data
        }
        
        # Collecter les résultats au fur et à mesure
        completed = 0
        total = len(files)
        
        for future in as_completed(future_to_index):
            try:
                result = future.result()
                rows.append(result)
                completed += 1
                
                # Mettre à jour la barre de progression
                progress_bar.progress(completed / total)
                status_text.text(f"Traitement terminé: {completed}/{total}")
                
            except Exception as e:
                st.error(f"Erreur lors du traitement: {str(e)}")
                completed += 1
                progress_bar.progress(completed / total)
    
    # Trier les résultats par numéro d'email pour maintenir l'ordre
    rows.sort(key=lambda x: x.get("Numéro de l'email par date", ""))
    
    df = pd.DataFrame(rows)
    st.success(f"Synthèse terminée! {len(files)} emails traités en parallèle.")
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
