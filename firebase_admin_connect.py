# firebase_admin_connect.py

import json
import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore

# Pull the JSON blob out of your secrets.toml
sa = json.loads(st.secrets["firebase"]["account_key"])

# Only initialize once
if not firebase_admin._apps:
    cred = credentials.Certificate(sa)
    firebase_admin.initialize_app(cred)

db = firestore.client()
