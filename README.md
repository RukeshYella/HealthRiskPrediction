
# 🏥 Hospital Risk Intelligence Dashboard

This project is a rule-based health risk scoring application designed for hospitals to **triage**, **monitor**, and **track** patient vitals in both emergency room (in-person) and remote (home) settings.

It provides real-time risk scoring, automated classification, historical trend tracking, and summary dashboards — all accessible through an interactive Dash web interface.

---

## 🚀 Features

✅ Real-time patient risk scoring calculator  
✅ Live tables for in-person (ER) and remote monitoring use cases  
✅ Color-coded risk summaries (Low, Medium, High)  
✅ Patient trend tracking over time  
✅ Exportable dashboards and summaries  
✅ Scalable for future integrations (alerts, role-based access, mobile interface)

---

## 🛠️ Technologies Used

- **Python 3.x**
- **Dash** (Plotly Dash framework)
- **Pandas, NumPy** (data processing)
- **Scikit-learn** (scaling & feature processing)
- **Joblib** (model saving/loading)
- **CSV** (local session storage)

---

## 📂 Project Structure

```
/scripts/
    app.py                     → Main Dash application  
    create_scoring_features.py → Script to generate feature weights
/models/
    scaler.pkl                 → Saved scaler for feature normalization
    scoring_features.pkl       → Saved feature configuration
/data/
    session_scores.csv         → Live session score log
    sample_60_cases.csv        → Sample cases used for weight derivation
```

---

## ⚙️ Setup Instructions

1️⃣ **Clone the repository**
```bash
git clone https://github.com/RukeshYella/HealthRiskPrediction
cd HealthRiskPrediction
```

2️⃣ **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3️⃣ **Install required packages**
```bash
pip install -r requirements.txt
```

4️⃣ **Generate feature files (if not present)**
```bash
python scripts/create_scoring_features.py
```

5️⃣ **Run the application**
```bash
python scripts/app.py
```

The app will launch locally at `http://127.0.0.1:8050/`.

---

## 🖥️ Application Pages

- **Home** → Overview and daily risk summary  
- **Risk Calculator** → Input patient vitals, get instant risk score + trend  
- **InPerson (ER) Page** → Table + graph of recent ER patient risks  
- **Remote Monitoring Page** → Table + graph of recent home monitoring risks

---

## 📊 Sample Screenshots

👉 (You will insert screenshots here showing Home page, Calculator, Tables, etc.)
![image](https://github.com/user-attachments/assets/ee0596ec-ad7d-4b93-914c-37631630cc52)
![image](https://github.com/user-attachments/assets/7134725b-6c84-477c-b2cb-2d7d51f56651)
![image](https://github.com/user-attachments/assets/7c27dddc-f16b-446d-b719-40a85150bf18)

---

## 🔒 Future Enhancements

- Role-based login (Admins, Nurses, Patients)  
- PDF/CSV export features  
- Mobile-friendly interface for patient-side use  
- Alert system for high-risk cases  
- Integration with live EHR systems

---
