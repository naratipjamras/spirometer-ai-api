import joblib
import pandas as pd

bundle = joblib.load("spirometry_model.joblib")
model = bundle["model"]
le = bundle["label_encoder"]
features = bundle["features"]

# ข้อมูลผู้ป่วยใหม่ (FEV1_PER_FVC = 2.0 / 3.8)
new_patient = pd.DataFrame([[55, 168, 72, 3.8, 2.0, 4.2, 2.0/3.8]], columns=features)

pred = model.predict(new_patient)[0]          # ได้เลข เช่น 1
print(le.inverse_transform([pred])[0])        # ได้ชื่อ เช่น "Obstructive"