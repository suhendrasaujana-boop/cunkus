import joblib

try:
    # Mencoba memuat file
    data = joblib.load('global_model.joblib')
    
    print("✅ File berhasil dibuka oleh Python!")
    print("Tipe data:", type(data))
    
    if isinstance(data, dict):
        print("Isi file adalah dictionary dengan keys:", data.keys())
        if 'model' in data:
            print("Model yang tersimpan:", data['model'])
        if 'feature_names' in data:
            print("Fitur yang digunakan:", data['feature_names'])
            
except Exception as e:
    print("❌ Gagal membuka file. Kemungkinan file kosong atau rusak.")
    print("Detail error:", e)
