from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn
import sys

# Inisialisasi App
app = FastAPI()

# --- LOAD MODEL ---
try:
    # Sesuaikan nama file dengan yang ada di folder 'model' kamu
    print("Sedang meload model...")
    model = joblib.load('model/model_nb.pkl') 
    vectorizer = joblib.load('model/vectorizer_tfidf.pkl')
    print("Model berhasil diload!")
except Exception as e:
    print(f"ERROR: Gagal load model. Pastikan file ada di folder 'model'. Detail: {e}")
    sys.exit(1) # Matikan program jika model tidak ketemu

# Schema Input
class TextInput(BaseModel):
    text: str

# Endpoint Prediksi
@app.post("/predict")
def predict_sentiment(input_data: TextInput):
    text = input_data.text
    
    # Preprocessing & Prediksi
    try:
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        
        # ... (kode atas sama) ...

        # --- UPDATE MAPPING EMOSI (Sesuai Abjad) ---
        label_map = {
            0: "Cemas 😰",
            1: "Marah 😡",
            2: "Netral 😐",
            3: "Optimis 🌟",
            4: "Sedih 😢",
            5: "Senang 😄"
        }
        
        # Ambil label teks berdasarkan angka prediksi
        # Int(prediction) dipakai karena output model kadang float (2.0)
        hasil_label = label_map.get(int(prediction), f"Emosi Lain ({prediction})")
        
        # -----------------------------------------

        return {
            "text": text,
            "prediction": hasil_label, 
            "status": "success"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)