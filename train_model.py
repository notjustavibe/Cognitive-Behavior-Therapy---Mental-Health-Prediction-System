"""
MindCheck model trainer.

Run once:   python train_model.py
It reads the big CSVs, trains the model and saves model.pkl.
After that the web app only loads model.pkl (tiny + fast) and never
touches the CSVs again. app.py calls ensure_model() automatically, so
if model.pkl is missing it is created on the very first run.
"""
import os, gc, datetime
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
COUNTRY_LIST  = ['United States','United Kingdom','Canada','Germany',
                 'Netherlands','Ireland','Australia','France','India','Other']

def load_and_merge():
    """Load both datasets, harmonise features, return combined DataFrame."""
    import pandas as pd  # imported lazily: not needed once model.pkl exists

    # ── OSMI Survey (DS1) ────────────────────────────────────────────────────
    df1 = pd.read_csv(os.path.join(BASE_DIR, "survey.csv"))

    def clean_gender(g):
        g = str(g).strip().lower()
        if g in ['male','m','man','cis male','male (cis)','mal','maile',
                 'male-ish','malr','msle','something kinda male?']: return 0
        elif g in ['female','f','woman','cis female','trans-female','femake',
                   'femail','female (trans)','queer/she/they','female ']: return 1
        return 2

    df1['gender_enc']  = df1['Gender'].apply(clean_gender)
    df1['age_clean']   = df1['Age'].apply(lambda x: x if 15<=x<=75 else np.nan).fillna(30)
    df1['age_bucket']  = df1['age_clean'].apply(
        lambda a: 0 if a<25 else (1 if a<35 else (2 if a<45 else 3)))
    df1['country_enc'] = df1['Country'].apply(
        lambda x: COUNTRY_LIST.index(x) if x in COUNTRY_LIST else 9)

    wi_f = {'Never':0.0,'Rarely':0.25,'Sometimes':0.6,'Often':1.0}
    df1['growing_stress']   = df1['work_interfere'].map(wi_f).fillna(0.5)
    df1['changes_habits']   = df1['mental_health_consequence'].map(
        {'No':0,'Maybe':0.5,'Yes':1}).fillna(0.5)
    df1['mh_history']       = 0.5
    df1['coping_struggles'] = df1['work_interfere'].map(
        {'Often':1,'Sometimes':0.6,'Rarely':0.2,'Never':0}).fillna(0.4)
    df1['work_interest']    = 1 - df1['work_interfere'].map(
        {'Often':0,'Sometimes':0.3,'Rarely':0.7,'Never':1}).fillna(0.5)
    df1['social_weakness']  = df1['coworkers'].map(
        {'Yes':0,'Some of them':0.5,'No':1}).fillna(0.5)
    df1['mh_interview']     = df1['mental_health_interview'].map(
        {'No':1,'Maybe':0.5,'Yes':0}).fillna(1)
    df1['care_options']     = df1['care_options'].map(
        {'No':2,'Not sure':1,'Yes':0}).fillna(1)
    df1['family_risk']      = (df1['family_history']=='Yes').astype(float)
    df1['has_treatment']    = (df1['treatment']=='Yes').astype(float)
    df1['mood_swings']      = df1['work_interfere'].map(
        {'Often':2,'Sometimes':1,'Rarely':0.5,'Never':0}).fillna(1)
    df1['days_indoors']     = df1['remote_work'].map({'Yes':3,'No':1}).fillna(1)
    df1['occupation']       = 0  # mostly corporate/tech

    # DS1 risk target
    wi_r = {'Never':1,'Rarely':2,'Sometimes':3,'Often':4}
    mhc_r= {'No':0,'Maybe':1,'Yes':2}
    lv_r = {'Very easy':0,'Somewhat easy':1,"Don't know":2,
             'Somewhat difficult':3,'Very difficult':4}
    sh_r = {'Yes':0,"Don't know":1,'No':2}
    bn_r = {'Yes':0,"Don't know":1,'No':2}
    df1['_ws'] = df1['work_interfere'].map(wi_r).fillna(2)
    df1['_sf'] = df1['mental_health_consequence'].map(mhc_r).fillna(1)
    df1['_ob'] = (df1['obs_consequence']=='Yes').astype(int)
    df1['_nb'] = df1['benefits'].map(bn_r).fillna(1)
    df1['_ld'] = df1['leave'].map(lv_r).fillna(2)
    df1['_sh'] = df1['seek_help'].map(sh_r).fillna(1)
    df1['_re'] = (df1['remote_work']=='Yes').astype(int)
    df1['risk_raw'] = (df1['_ws'] + df1['family_risk'] + df1['has_treatment'] +
                       df1['_sf'] + df1['_ob']*0.5 + df1['_nb']*0.5 +
                       df1['_ld']*0.4 + df1['_sh']*0.5 + df1['_re']*0.3)
    df1['risk_level'] = pd.cut(df1['risk_raw'],
        bins=[0,3.5,5.5,7.5,100], labels=[0,1,2,3]).astype(int)

    # ── Mental Health Dataset (DS2) ──────────────────────────────────────────
    # Sample FIRST (same rows as before: same seed, same index) so the heavy
    # feature engineering below runs on 30k rows instead of 292k -> far less RAM.
    df2 = pd.read_csv(os.path.join(BASE_DIR, "Mental_Health_Dataset.csv"))
    df2 = df2.sample(30000, random_state=42).reset_index(drop=True)

    yn = {'Yes':1,'No':0,'Maybe':0.5}
    df2['growing_stress']   = df2['Growing_Stress'].map(yn).fillna(0.5)
    df2['changes_habits']   = df2['Changes_Habits'].map(yn).fillna(0.5)
    df2['mh_history']       = df2['Mental_Health_History'].map(yn).fillna(0.5)
    df2['coping_struggles'] = df2['Coping_Struggles'].map({'Yes':1,'No':0}).fillna(0)
    df2['work_interest']    = df2['Work_Interest'].map(
        {'Yes':0,'Maybe':0.5,'No':1}).fillna(0.5)
    df2['social_weakness']  = df2['Social_Weakness'].map(yn).fillna(0.5)
    df2['mh_interview']     = df2['mental_health_interview'].map(
        {'Yes':0,'Maybe':0.5,'No':1}).fillna(1)
    df2['care_options']     = df2['care_options'].map(
        {'Yes':0,'Not sure':1,'No':2}).fillna(1)
    df2['family_risk']      = (df2['family_history']=='Yes').astype(float)
    df2['has_treatment']    = (df2['treatment']=='Yes').astype(float)
    df2['mood_swings']      = df2['Mood_Swings'].map(
        {'Low':0,'Medium':1,'High':2}).fillna(1)
    df2['days_indoors']     = df2['Days_Indoors'].map(
        {'Go out Every day':0,'1-14 days':1,'15-30 days':2,
         '31-60 days':3,'More than 2 months':4}).fillna(2)
    df2['occupation']       = df2['Occupation'].map(
        {'Corporate':0,'Business':1,'Student':2,'Housewife':3,'Others':4}).fillna(4)
    df2['gender_enc']       = df2['Gender'].map({'Male':0,'Female':1}).fillna(0).astype(int)
    df2['country_enc']      = df2['Country'].apply(
        lambda x: COUNTRY_LIST.index(x) if x in COUNTRY_LIST else 9)
    df2['age_bucket']       = 1  # not available

    def r2(row):
        s  = row['growing_stress']*2 + row['coping_struggles']*2
        s += row['mood_swings'] + row['social_weakness'] + row['work_interest']
        s += row['changes_habits'] + row['mh_history']
        s += row['days_indoors']*0.5 + row['care_options']*0.3 + row['family_risk']*0.5
        return s
    df2['risk_raw']   = df2.apply(r2, axis=1)
    df2['risk_level'] = pd.cut(df2['risk_raw'],
        bins=[0,2.5,5.0,7.5,100], labels=[0,1,2,3]).astype(int)

    # ── Interaction features ─────────────────────────────────────────────────
    def interactions(df):
        df = df.copy()
        df['stress_coping']  = df['growing_stress'] * df['coping_struggles']
        df['mood_social']    = df['mood_swings']    * df['social_weakness']
        df['indoor_stress']  = df['days_indoors']   * df['growing_stress']
        df['care_history']   = df['care_options']   * df['mh_history']
        return df

    df1 = interactions(df1)
    df2 = interactions(df2)

    # ── Combine ──────────────────────────────────────────────────────────────
    COLS = BASE_COLS + INTERACTION_COLS
    ds2_sample = df2
    ds1_rep    = pd.concat([df1]*8, ignore_index=True)   # balance DS1
    combined   = pd.concat(
        [ds1_rep[COLS+['risk_level']], ds2_sample[COLS+['risk_level']]],
        ignore_index=True).dropna()
    return combined

BASE_COLS = [
    'growing_stress','changes_habits','mh_history','coping_struggles',
    'work_interest','social_weakness','mh_interview','care_options',
    'family_risk','has_treatment','mood_swings','days_indoors',
    'gender_enc','country_enc','occupation','age_bucket'
]
INTERACTION_COLS = ['stress_coping','mood_social','indoor_stress','care_history']
ALL_COLS = BASE_COLS + INTERACTION_COLS

# Lighter forest than the original (400 trees, depth 20, leaf 1 -> 200 MB).
RF_PARAMS = dict(n_estimators=100, max_depth=20, min_samples_leaf=1,
                 max_features='sqrt', random_state=42, n_jobs=1)


def train_and_save(path=MODEL_PATH):
    print("[MindCheck] Training model from CSVs (first run only)...")
    combined = load_and_merge()
    X = combined[ALL_COLS].values.astype(np.float32)
    y = combined['risk_level'].values
    del combined; gc.collect()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    acc = round(accuracy_score(y_test, model.predict(X_test)) * 100, 2)
    joblib.dump({"model": model, "accuracy": acc, "columns": ALL_COLS,
                 "trained_at": datetime.datetime.now().isoformat(timespec="seconds")},
                path, compress=3)
    print(f"[MindCheck] Saved {os.path.basename(path)} — accuracy {acc}%")
    return model, acc


def ensure_model(path=MODEL_PATH):
    """Load model.pkl if it exists, otherwise train once and create it."""
    if os.path.exists(path):
        try:
            blob = joblib.load(path)
            if blob.get("columns") == ALL_COLS:
                return blob["model"], blob["accuracy"]
            print("[MindCheck] model.pkl is outdated -> retraining")
        except Exception as e:
            print(f"[MindCheck] Could not read model.pkl ({e}) -> retraining")
    return train_and_save(path)


if __name__ == "__main__":
    train_and_save()
