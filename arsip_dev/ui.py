# file: ui.py
import streamlit as st
import requests

st.title("Analisis Sentiment Teks")
st.write("Masukkan teks di bawah ini untuk dianalisis oleh AI.")

# 1. Input Teks dari User
user_input = st.text_area("Masukkan Teks/Kalimat:", height=150)

# 2. Tombol Prediksi
if st.button("Analisis"):
    if user_input:
        # Tampilkan loading spinner
        with st.spinner('Sedang memproses...'):
            try:
                # 3. Kirim data ke API FastAPI
                # Alamat API lokal kita
                api_url = "http://127.0.0.1:8000/predict"
                payload = {"text": user_input}
                
                response = requests.post(api_url, json=payload)
                
                # 4. Tampilkan Hasil
                if response.status_code == 200:
                    result = response.json()
                    prediction = result['prediction']
                    
                    st.success("Selesai!")
                    st.metric(label="Hasil Prediksi", value=prediction)
                    
                    # Tampilkan raw json jika perlu untuk debug
                    with st.expander("Lihat Detail JSON"):
                        st.json(result)
                else:
                    st.error("Gagal menghubungi API.")
                    
            except Exception as e:
                st.error(f"Terjadi kesalahan koneksi: {e}")
    else:
        st.warning("Mohon isi teks terlebih dahulu.")