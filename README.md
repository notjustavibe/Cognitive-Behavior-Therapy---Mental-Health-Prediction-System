# MindCheck Final Version — Cognitive Behavioral Report
## Qasim's MindCheck Model

### Improvements Made Over The Beta Version

| Feature | Beta | Final |
|---|---|---|
| Datasets | OSMI survey (1,259 rows) | OSMI + Mental Health Dataset (293K+ rows) |
| Model accuracy | 88% | 98% |
| Interaction features | ❌ | ✅ stress×coping, mood×social, etc. |
| Report title | Generic | "Cognitive Behavioral Report" |
| Patient name | ❌ | ✅ Appears on report header |
| UI design | Minimal | Editorial clinical letterhead |
| Days indoors feature | ❌ | ✅ (from DS2) |
| Mood swings feature | ❌ | ✅ (from DS2) |
| Occupation feature | ❌ | ✅ (Corporate/Student/etc.) |

### Run locally

```bash
pip install -r requirements.txt
python app.py          # first run trains the model and creates model.pkl (~3 s)
# Open http://localhost:5000
```

`model.pkl` (~9 MB) is created automatically on the first run. Every later run just
loads it — the big CSVs are not read again. To retrain, delete `model.pkl` (or run
`python train_model.py`).

### Deploying (Render / PythonAnywhere)
Commit `model.pkl` to the repo. The server then never needs the CSVs, so you can
leave `Mental_Health_Dataset.csv` out of the deployment.
- Render start command: `gunicorn app:app --workers 1`
- PythonAnywhere: point the WSGI file at `from app import app as application`

### Data sources
1. OSMI Mental Health in Tech Survey (2014) — 1,259 responses
2. Mental Health Dataset (Kaggle) — 292,364 responses

### Features used by model
- Growing stress (work/academic pressure)
- Changes in habits (behavioral shifts)
- Mental health history
- Coping struggles
- Work/study interest
- Social weakness / isolation
- Mental health interview willingness
- Care options availability
- Family risk history
- Treatment history
- Mood swings
- Days spent indoors
- Gender, country, occupation, age group
- Interaction terms: stress×coping, mood×social, indoor×stress, care×history
