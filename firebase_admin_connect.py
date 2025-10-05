# firebase_admin_connect.py  – safe single‑initialisation wrapper

import firebase_admin
from firebase_admin import credentials, firestore

# prevent “default Firebase app already exists” on repeated imports
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()
