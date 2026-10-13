from flask import Flask, render_template, request
import numpy as np
import warnings, datetime
warnings.filterwarnings('ignore')

from train_model import ensure_model, COUNTRY_LIST

app = Flask(__name__)

CLINIC_NAME   = "Qasim's MindCheck Model"
REPORT_TITLE  = "Cognitive Behavioral Report"

# ── Model: loaded from model.pkl; trained + saved automatically on first run ──
MODEL, MODEL_ACCURACY = ensure_model()
print(f"[MindCheck] Model ready — Accuracy: {MODEL_ACCURACY}%")


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

RISK_LABELS = ['Normal', 'Mild', 'Moderate', 'Severe']
RISK_COLORS = ['#2A9E6E', '#E8A020', '#D05020', '#9E1C1C']
RISK_BG     = ['#E8F8F0', '#FEF3E2', '#FCEAE4', '#FAE8E8']

DOMAIN_REC = {
    'emotional': {
        0: "Emotional equilibrium is strong. Sustain it through regular reflection, quality sleep, and social engagement.",
        1: "Occasional low moods or anxious thoughts are present. Brief daily mindfulness or journalling can help stabilise.",
        2: "Noticeable emotional strain detected. A conversation with a counsellor or trusted person is strongly advisable.",
        3: "Significant emotional distress indicated. Please reach out to a mental health professional — support is available.",
    },
    'social': {
        0: "Social connections are healthy and supportive. Keep nurturing these bonds — they are a protective factor.",
        1: "Mild social withdrawal present. Scheduling regular low-pressure contact with others can help.",
        2: "Moderate isolation risk detected. Joining a group, class, or community activity can rebuild connection.",
        3: "High isolation risk. Human connection is critical right now — please reach out to someone you trust today.",
    },
    'stress': {
        0: "Stress and workload feel manageable. Maintain current boundaries and healthy coping routines.",
        1: "Moderate stress present. Prioritise tasks, take structured breaks, and protect sleep hours.",
        2: "High stress affecting functioning. Consider speaking to a counsellor about workload and pressure management.",
        3: "Overwhelming stress levels. Seek academic or workplace support and reduce non-essential commitments urgently.",
    },
    'outlook': {
        0: "A positive and grounded outlook on the future. Continue setting small, meaningful goals.",
        1: "Mild uncertainty about the future. Break larger goals into achievable milestones for clarity.",
        2: "Future feels uncertain or threatening. Career or goal-setting guidance can help create a clearer path.",
        3: "Severe future insecurity detected. A mentor, counsellor, or advisor can help you find direction and hope.",
    },
}

OVERALL_REC = {
    0: [
        "Maintain consistent sleep, exercise, and social routines.",
        "Continue hobbies and activities that bring you meaning.",
        "Perform brief monthly self-check-ins to track any changes.",
    ],
    1: [
        "Introduce 10 minutes of daily mindfulness or breathing exercises.",
        "Target 7–8 hours of sleep each night consistently.",
        "Talk openly with a trusted friend or family member.",
        "Reduce passive screen time, especially before bed.",
    ],
    2: [
        "Schedule a consultation with a counsellor or therapist.",
        "Reduce academic or work commitments where feasible.",
        "Engage in physical activity at least 3 times per week.",
        "Avoid relying on caffeine, alcohol, or substances to cope.",
        "Explore mental health resources available in your institution or workplace.",
    ],
    3: [
        "Seek professional mental health support as a matter of priority.",
        "Inform a trusted person — family, friend, or manager — about how you are feeling.",
        "Contact a mental health helpline if you feel in crisis.",
        "Take medical leave if required; your health must come first.",
        "Actively resist isolation — stay connected with your support network.",
    ],
}

COUNTRY_RESOURCES = {
    'United States':  "SAMHSA helpline: 1-800-662-4357. Employee Assistance Programmes (EAPs) are available at most workplaces.",
    'United Kingdom': "NHS Talking Therapies and Mind UK (mind.org.uk) offer free mental health support.",
    'Canada':         "Crisis Services Canada: 1-833-456-4566. Provincial health plans cover counselling.",
    'Germany':        "German statutory health insurance (GKV) covers psychotherapy. Ask your GP for a referral.",
    'Australia':      "Beyond Blue (beyondblue.org.au) and Lifeline (13 11 14) offer free national support.",
    'Netherlands':    "GGZ care is available through Dutch health insurance. Contact your huisarts (GP) for referral.",
    'Ireland':        "Pieta House (116 123) and HSE mental health services are available nationally.",
    'France':         "Numéro National de Prévention du Suicide: 3114. CMPs (mental health centres) are free.",
    'India':       "Visit openwho.org or contact a local GP. The WHO Mental Health Atlas lists services globally.",
    'Other':          "For immediate mental health emergencies, call the Umang Pakistan 24/7 Helpline at 0311-7786264. For general medical emergencies, dial 1122 (Rescue 1122) nationwide.",
}

AGE_INSIGHTS = {
    'Under 25': "Young adults experience peak onset of many mental health conditions. Early support leads to significantly better outcomes.",
    '25–34':    "Early-career pressure, financial stress, and identity transitions are common drivers in this age group.",
    '35–44':    "Mid-life stress often stems from compounded responsibilities. Work-life balance is a key intervention point.",
    '45+':      "Greater life experience offers resilience, but social isolation risk can increase. Regular connection is protective.",
}

GENDER_NOTES = {
    'Male':   "Men statistically under-report distress and are less likely to seek help — yet early intervention has strong positive outcomes. Reaching out is a sign of strength.",
    'Female': "Women tend to seek support more readily, which is strongly associated with better recovery. Your self-awareness is a genuine asset.",
    'Other':  "Non-binary and gender-diverse individuals often face compounding stressors. Affirming, inclusive mental health care is available and effective.",
}

def score_to_10(raw, invert=False):
    norm = ((raw - 1) / 4) * 10
    return round(10 - norm if invert else norm, 1)

def to_risk(score, lo=(1.5, 2.5, 3.5)):
    if score <= lo[0]: return 0
    elif score <= lo[1]: return 1
    elif score <= lo[2]: return 2
    return 3

def build_feature_vector(form, gender_enc, age_bucket, country_enc):
    """Map user's form inputs → model feature vector."""
    dep  = int(form['depression'])
    anx  = int(form['anxiety'])
    iso  = int(form['isolation'])
    soc  = int(form['social_relationships'])
    pres = int(form['academic_pressure'])
    fin  = int(form['financial_concerns'])
    sat  = int(form['study_satisfaction'])
    slp  = int(form['average_sleep'])
    spt  = int(form['sports_engagement'])
    disc = int(form['campus_discrimination'])
    fam  = int(form['family_history'])
    hs   = int(form['help_seeking'])        # 1=likely,2=maybe,3=unlikely
    stg  = int(form['stigma_concern'])      # 1-5
    ms   = int(form['mood_swings_q'])       # 1=low,2=med,3=high
    di   = int(form['days_indoors_q'])      # 1-5
    occ  = int(form['occupation'])          # 0-4

    # Map to feature space
    growing_stress   = (dep/5 + anx/5 + pres/5) / 3
    changes_habits   = stg / 5
    mh_history       = fam * 0.8
    coping_struggles = (dep/5 * 0.5 + (1 - sat/5)*0.3 + fin/5*0.2)
    work_interest    = (1 - sat/5)*0.6 + fin/5*0.2 + pres/5*0.2
    social_weakness  = iso/5 * 0.6 + (1 - soc/5)*0.4
    mh_interview     = (hs - 1) / 2          # 0=seeks,1=maybe,2=avoids
    care_options_f   = (hs - 1)               # 0,1,2
    family_risk      = float(fam)
    has_treatment    = 1.0 if hs == 1 else 0.0
    mood_swings      = float(ms - 1)          # 0,1,2
    days_indoors     = float(di - 1)          # 0-4

    # Interaction features
    stress_coping  = growing_stress * coping_struggles
    mood_social    = mood_swings    * social_weakness
    indoor_stress  = days_indoors   * growing_stress
    care_history   = care_options_f * mh_history

    return np.array([[
        growing_stress, changes_habits, mh_history, coping_struggles,
        work_interest, social_weakness, mh_interview, care_options_f,
        family_risk, has_treatment, mood_swings, days_indoors,
        gender_enc, country_enc, occ, age_bucket,
        stress_coping, mood_social, indoor_stress, care_history
    ]], dtype=float)


def predict(form):
    # Demographic inputs
    gender_map  = {'Male':0,'Female':1,'Other':2}
    age_map     = {'Under 25':0,'25–34':1,'35–44':2,'45+':3}

    gender_str  = form.get('gender','Male')
    age_str     = form.get('age_group','25–34')
    country_str = form.get('country','Other')
    patient_name= form.get('patient_name','Anonymous').strip() or 'Anonymous'

    gender_enc  = gender_map.get(gender_str, 0)
    age_bucket  = age_map.get(age_str, 1)
    country_enc = COUNTRY_LIST.index(country_str) if country_str in COUNTRY_LIST else 9

    X_input = build_feature_vector(form, gender_enc, age_bucket, country_enc)

    pred_class = int(MODEL.predict(X_input)[0])
    proba      = MODEL.predict_proba(X_input)[0].tolist()

    # Domain raw scores for display (1-5 scale)
    dep = int(form['depression'])
    anx = int(form['anxiety'])
    iso = int(form['isolation'])
    soc = int(form['social_relationships'])
    pres= int(form['academic_pressure'])
    fin = int(form['financial_concerns'])
    sat = int(form['study_satisfaction'])

    emo_raw  = (dep + anx) / 2
    soc_raw  = (iso + (6 - soc)) / 2
    str_raw  = (pres + fin + (6 - sat)) / 3
    out_raw  = int(form['future_insecurity'])
    comp_raw = (emo_raw + soc_raw + str_raw + out_raw) / 4

    domains = {}
    for key, label, val in [
        ('emotional', 'Emotional Health',   emo_raw),
        ('social',    'Social Wellbeing',    soc_raw),
        ('stress',    'Stress Resilience',   str_raw),
        ('outlook',   'Future Outlook',      out_raw),
    ]:
        r = to_risk(val, lo=(2.0, 3.0, 4.0))
        domains[key] = {
            'label': label,
            'score': round(score_to_10(val), 1),
            'risk': r,
            'risk_label': RISK_LABELS[r],
            'color': RISK_COLORS[r],
            'bg':    RISK_BG[r],
            'recommendation': DOMAIN_REC[key][r],
        }

    country_resource = COUNTRY_RESOURCES.get(country_str, COUNTRY_RESOURCES['Other'])
    family_note = (
        "Family history of mental illness is a documented risk factor. "
        "Regular check-ins with a healthcare professional are advisable."
        if int(form.get('family_history', 0)) == 1 else ""
    )

    recs = list(OVERALL_REC[pred_class])
    recs.append(f"📍 {country_str} resource: {country_resource}")

    now = datetime.datetime.now()
    report_date = f"{now.day} {now.strftime('%B %Y')}"

    return {
        'patient_name':   patient_name,
        'report_date':    report_date,
        'clinic_name':    CLINIC_NAME,
        'report_title':   REPORT_TITLE,
        'overall_risk':   pred_class,
        'overall_label':  RISK_LABELS[pred_class],
        'overall_color':  RISK_COLORS[pred_class],
        'overall_bg':     RISK_BG[pred_class],
        'overall_score':  round(score_to_10(comp_raw), 1),
        'confidence':     round(max(proba) * 100, 1),
        'proba':          {RISK_LABELS[i]: round(proba[i]*100,1) for i in range(4)},
        'domains':        domains,
        'recommendations': recs,
        'model_accuracy':  MODEL_ACCURACY,
        'demo': {
            'gender':       gender_str,
            'age_group':    age_str,
            'country':      country_str,
            'gender_note':  GENDER_NOTES.get(gender_str, ''),
            'age_insight':  AGE_INSIGHTS.get(age_str, ''),
            'family_note':  family_note,
            'country_note': country_resource,
        }
    }

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html',
                           clinic_name=CLINIC_NAME,
                           report_title=REPORT_TITLE,
                           model_accuracy=MODEL_ACCURACY)

@app.route('/predict', methods=['POST'])
def predict_route():
    result = predict(request.form)
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
