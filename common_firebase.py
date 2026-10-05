            db.collection("detections").add(data)
            st.success("Detection result stored in Firebase!")
        except Exception as firestore_error:
            st.error(f"Error storing detection data: {firestore_error}")

# ─────────────────────────────  UI PARTS  ───────────────────────────────
def auth_sidebar():
    st.sidebar.header("Authentication")
    act   = st.sidebar.radio("Action",["Login","Sign Up"])
    email = st.sidebar.text_input("Email")
    pwd   = st.sidebar.text_input("Password",type="password")
    if act=="Login" and st.sidebar.button("Login"):
        try:
            u=auth.sign_in_with_email_and_password(email,pwd)
            st.session_state.update(user_email=email,idToken=u["idToken"],uid=u["localId"],show_auth=False); st.rerun()
        except Exception as e: st.error(f"Login failed: {e}")
    if act=="Sign Up" and st.sidebar.button("Sign Up"):
        try:
            u=auth.create_user_with_email_and_password(email,pwd)
            st.session_state.update(user_email=email,idToken=u["idToken"],uid=u["localId"],show_auth=False); st.rerun()
        except Exception as e: st.error(f"Signup failed: {e}")

def welcome_interface():
    st.markdown("<h1 style='text-align:center'>AVIA</h1>"
                "<p style='text-align:center;margin-top:-1rem;font-size:1.3rem'>"
                "Audio-Visual Integrity Analyzer&nbsp;|&nbsp;Deepfake Detection</p><hr>",
                unsafe_allow_html=True)
    _,mid,_ = st.columns([1,2,1])
    with mid:
        l,r = st.columns(2)
        if l.button("🔓 Guest Mode"):
            st.session_state.update(user_email="guest",guest_count=5,show_auth=False); st.rerun()
        if r.button("🔐 Login / Sign Up"): st.session_state.show_auth=True
    if st.session_state.show_auth: auth_sidebar()

def detection_interface():
    st.title("AVIA")
    mode = st.selectbox("Choose Detection Type",("Audio","Image","Video"),index=0)

    # 1️⃣ upload widget (always visible, translucent background)
    if mode=="Audio":
        uploaded = st.file_uploader("Upload Audio",type=["wav","mp3","ogg"])
    elif mode=="Image":
        uploaded = st.file_uploader("Upload Image",type=["jpg","jpeg","png"])
    else:
        uploaded = st.file_uploader("Upload Video",type=["avi","mp4","mov","mpg","mpeg"])

    # 2️⃣ card shows ONLY once a file is selected
    if uploaded:
        st.markdown("<div class='card'>",unsafe_allow_html=True)

        # AUDIO
        if mode=="Audio":
            if st.button("Analyze Audio"):
                pred = load_audio_model().predict(extract_features(uploaded.getvalue()))[0][0]
                lbl,conf = ("Fake",pred*100) if pred>0.5 else ("Real",(1-pred)*100)
                st.success(f"{lbl} • {conf:.2f}%"); store_result("Audio",uploaded.name,lbl,conf)

        # IMAGE
        elif mode=="Image":
            img=Image.open(uploaded).convert("RGB"); st.image(img)
            if st.button("Analyze Image"):
                dev=torch.device("cpu")
                mdl=convnext_image_model().to(dev)
                mdl.load_state_dict(torch.load("checkpoint_epoch_20 (2).pth",map_location=dev)["model_state_dict"])
                lbl,conf=test_single(mdl,img,dev); st.success(f"{lbl} • {conf:.2f}%"); store_result("Image",uploaded.name,lbl,conf)
                if lbl=="Fake":
                    mdl2=convnext_image_model_tech().to(dev)
                    mdl2.load_state_dict(torch.load("convnext2_epoch_20.pth",map_location=dev)["model_state_dict"])
                    tech,tconf=test_tech(mdl2,img,dev); st.info(f"Technology: {tech} ({tconf:.2f}%)")

        # VIDEO
        else:
            st.video(uploaded)
            frames = st.number_input("Frames to sample",1,60,15)
            fp16   = st.checkbox("Enable FP16",True) if torch.cuda.is_available() else False
            if st.button("Analyze Video"):
                tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4"); tmp.write(uploaded.getvalue()); tmp.close()
                cfg=load_config(); remove_key_recursively(cfg,"pretrained_cfg")
                mdl=load_genconvit(cfg,net="genconvit",ed_weight="genconvit_ed_inference",
                                   vae_weight="genconvit_vae_inference",fp16=fp16)
                if fp16: mdl=mdl.half()
                y,conf=predict_video_file(tmp.name,mdl,frames,fp16=fp16); os.remove(tmp.name)
                if y is not None:
                    pred=real_or_fake(y); conf=1-conf if pred=="REAL" else conf
                    st.success(f"{pred} • {conf*100:.2f}%"); store_result("Video",uploaded.name,pred,conf*100)
                else:
                    st.error("Prediction failed")

        st.markdown("</div>",unsafe_allow_html=True)  # close card





#----------------------------------------------------------------------------------
def load_history(user_email):
        docs = db.collection("detections").where("user", "==", user_email).stream()
        history_data = []
        for doc in docs:
            record = doc.to_dict()
            history_data.append({
                "UID": record.get("uid", ""),
                "User": record.get("user", ""),
                "File": record.get("filename", ""),
                "Type": record.get("type", ""),
                "Result": record.get("result", ""),
                "Confidence": record.get("confidence", ""),
                "Media URL": record.get("media_url", ""),
                "Timestamp": record.get("timestamp", "")
            })
        return pd.DataFrame(history_data)

# ─────────────────────────────  ROUTING  ───────────────────────────────
def main():
    if st.session_state.user_email is None:
        welcome_interface()

    elif st.session_state.user_email and st.session_state.user_email != "guest":
        # Logged‐in user sidebar actions
        with st.sidebar:
            if st.button("📂 View My History"):
                st.experimental_rerun()
            if st.button("🚪 Logout"):
                st.session_state.update(
                    user_email=None,
                    idToken=None,
                    uid=None,
                    show_auth=None
                )
                st.experimental_rerun()

        # Main detection interface
        detection_interface()

        # Then display the user’s history below
        history_df = load_history(st.session_state.user_email)
        if not history_df.empty:
            st.dataframe(history_df)
        else:
            st.info("No records found in your history yet.")

    else:
        # Guest mode (or anything else without a real account)
        detection_interface()

if __name__ == "__main__":
    main()
