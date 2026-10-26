import os, secrets, urllib.parse
from flask import (Flask, render_template, request, redirect, url_for,
                   session, flash, abort, g)
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import warnings, datetime
warnings.filterwarnings('ignore')

from train_model import ensure_model, COUNTRY_LIST
import db

app = Flask(__name__)

CLINIC_NAME   = "Qasim's MindCheck Model"
REPORT_TITLE  = "Cognitive Behavioral Report"
HELP_EMAIL    = "mq.faq1@gmail.com"

# ── Secret key (signs the login cookie). Set SECRET_KEY on the server; if it is
#    missing we generate one once and keep it in a file next to the database. ──
def _load_secret_key():
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    path = os.path.join(os.path.dirname(db.DB_PATH), ".secret_key")
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        key = secrets.token_hex(32)
        try:
            with open(path, "w") as f:
                f.write(key)
        except OSError:
            pass
        return key

app.config.update(
    SECRET_KEY=_load_secret_key(),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.environ.get("RENDER")),   # https on Render
)
db.init_db()

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
    'Pakistan':       "Emergency and mental health helpline numbers for Pakistan are listed below.",
    'Other':          "In an emergency, call your local emergency number. findahelpline.com lists free mental health helplines by country.",
}

# Pakistan numbers, checked against TherapyRoute (Pakistan crisis lines) and
# Zameen.com's emergency services list. Hours/coverage can change - the note is shown to users.
PAKISTAN_EMERGENCY = [
    ("Police (Madadgar)",             "15",               "Nationwide emergency"),
    ("Rescue 1122",                   "1122",             "Ambulance and rescue; availability depends on your province/city"),
    ("Edhi Ambulance",                "115",              "Charity ambulance service, widely available"),
    ("Chhipa Ambulance",              "1020",             "Charity ambulance service"),
    ("Sehat Tahaffuz Helpline",       "1166",             "National health helpline; its Humraaz service connects you to mental health counsellors"),
    ("Rozan Counselling Helpline",    "0800-22-444",      "Toll-free (also 0304-111-1741). Mon-Fri, 9:30 am-5:00 pm"),
    ("Taskeen Health Initiative",     "0316-827-5336",    "Mon-Sat, 11 am-11 pm. Free support, not a diagnosis"),
]
PAKISTAN_EMERGENCY_NOTE = (
    "Pakistan has no round-the-clock national mental health line. If you are in immediate "
    "danger, call 15, 1122 or 115, or go to the nearest hospital emergency department."
)

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

def build_feature_vector(form, gender_enc, age_bucket):
    """Map user's form inputs → model feature vector."""
    dep  = int(form['depression'])
    anx  = int(form['anxiety'])
    iso  = int(form['isolation'])
    soc  = int(form['social_relationships'])
    pres = int(form['academic_pressure'])
    fin  = int(form['financial_concerns'])
    sat  = int(form['study_satisfaction'])
    fam  = int(form['family_history'])
    hs   = int(form['help_seeking'])        # 1=likely,2=maybe,3=unlikely
    stg  = int(form['stigma_concern'])      # 1-5
    ms   = int(form['mood_swings_q'])       # 1=low,2=med,3=high
    di   = int(form['days_indoors_q'])      # 1-5
    occ  = int(form['occupation'])          # 0-4

    # Map to feature space.
    # IMPORTANT: 1-5 answers are rescaled with n() so that 1 -> 0.0 and 5 -> 1.0.
    # (The old x/5 gave 0.2 for the lowest answer, which silently added a
    #  baseline of "risk" for everyone and pushed healthy users into Mild.)
    n = lambda v: (v - 1) / 4.0

    growing_stress   = (n(dep) + n(anx) + n(pres)) / 3
    changes_habits   = n(stg)
    mh_history       = fam * 0.8
    coping_struggles = n(dep)*0.5 + (1 - n(sat))*0.3 + n(fin)*0.2
    work_interest    = (1 - n(sat))*0.6 + n(fin)*0.2 + n(pres)*0.2
    social_weakness  = n(iso)*0.6 + (1 - n(soc))*0.4
    mh_interview     = (hs - 1) / 2          # 0=seeks,1=maybe,2=avoids
    care_options_f   = (hs - 1)               # 0,1,2
    family_risk      = float(fam)
    has_treatment    = 0.0   # the form never asks about current treatment
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
        gender_enc, occ, age_bucket,
        stress_coping, mood_social, indoor_stress, care_history
    ]], dtype=float)


def predict(form, default_name='Anonymous'):
    # Demographic inputs
    gender_map  = {'Male':0,'Female':1,'Other':2}
    age_map     = {'Under 25':0,'25–34':1,'35–44':2,'45+':3}

    gender_str  = form.get('gender','Male')
    age_str     = form.get('age_group','25–34')
    country_str = form.get('country','Other')
    patient_name= (form.get('patient_name') or '').strip()[:80] or default_name

    gender_enc  = gender_map.get(gender_str, 0)
    age_bucket  = age_map.get(age_str, 1)

    X_input = build_feature_vector(form, gender_enc, age_bucket)

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
    # Overall score (0-10) = model's expected risk level, so it always agrees
    # with the predicted label (previously it used only 8 of the 20 inputs).
    overall_score10 = round(sum(i * p for i, p in enumerate(proba)) / 3 * 10, 1)

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

    # Pakistan selected at the start -> show Pakistani emergency / helpline numbers
    emergency = None
    if country_str == 'Pakistan':
        emergency = {
            'country': 'Pakistan',
            'numbers': [{'name': n, 'number': num, 'note': note}
                        for n, num, note in PAKISTAN_EMERGENCY],
            'note': PAKISTAN_EMERGENCY_NOTE,
        }

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
        'overall_score':  overall_score10,
        'confidence':     round(max(proba) * 100, 1),
        'proba':          {RISK_LABELS[i]: round(proba[i]*100,1) for i in range(4)},
        'domains':        domains,
        'recommendations': recs,
        'emergency':       emergency,
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
#  FORM VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

GENDERS    = ['Male', 'Female', 'Other']
AGE_GROUPS = ['Under 25', '25–34', '35–44', '45+']

# field -> allowed values.  REQUIRED questions (marked * on the page) must be
# answered; OPTIONAL ones (marked "optional") may be skipped.
REQUIRED_CHOICES = {
    'gender': GENDERS, 'age_group': AGE_GROUPS, 'country': COUNTRY_LIST,
    'occupation': ['0','1','2','3','4'], 'family_history': ['0','1'],
    'help_seeking': ['1','2','3'], 'days_indoors_q': ['1','2','3','4','5'],
    'mood_swings_q': ['1','2','3'], 'stigma_concern': list('12345'),
    'depression': list('12345'), 'anxiety': list('12345'),
    'isolation': list('12345'), 'social_relationships': list('12345'),
    'academic_pressure': list('12345'), 'financial_concerns': list('12345'),
    'study_satisfaction': list('12345'), 'future_insecurity': list('12345'),
}
OPTIONAL_CHOICES = {
    'campus_discrimination': ['0','1'], 'average_sleep': ['1','2','3'],
    'sports_engagement': ['1','2','3','4'],
}


def validate_assessment(form):
    """Returns a list of problems (empty list = OK)."""
    problems = []
    for field, allowed in REQUIRED_CHOICES.items():
        if form.get(field) not in allowed:
            problems.append(field)
    for field, allowed in OPTIONAL_CHOICES.items():
        if form.get(field, '') not in allowed + ['']:
            problems.append(field)
    return problems


def clean_answers(form):
    """What we store in the database: only the known questions, nothing else."""
    keys = ['patient_name'] + list(REQUIRED_CHOICES) + list(OPTIONAL_CHOICES)
    return {k: (form.get(k) or '').strip()[:80] for k in keys}


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION / CSRF / TEMPLATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def csrf_token():
    if '_csrf' not in session:
        session['_csrf'] = secrets.token_hex(16)
    return session['_csrf']


def help_url():
    """Opens a pre-addressed Gmail compose window to the help inbox."""
    q = urllib.parse.urlencode({
        'view': 'cm', 'fs': '1', 'to': HELP_EMAIL,
        'su': 'MindCheck - I would like some help',
        'body': 'Hello,\n\nI used MindCheck and would like some help.\n\n'
                '(Please write what you are comfortable sharing below.)\n\n',
    }, quote_via=urllib.parse.quote)
    return 'https://mail.google.com/mail/?' + q


@app.before_request
def load_user_and_check_csrf():
    uid = session.get('uid')
    g.user = db.get_user(uid) if uid else None
    if uid and g.user is None:          # account was deleted
        session.pop('uid', None)
    if request.method == 'POST':
        sent = request.form.get('csrf_token', '')
        if not sent or not secrets.compare_digest(sent, session.get('_csrf', '')):
            abort(400, 'Your session expired. Please go back, refresh the page and try again.')


@app.context_processor
def inject_globals():
    return dict(user=g.get('user'), csrf_token=csrf_token, help_email=HELP_EMAIL,
                clinic_name=CLINIC_NAME)


def login_required(view):
    from functools import wraps
    @wraps(view)
    def wrapped(*a, **kw):
        if g.user is None:
            flash('Please log in to see your profile.', 'info')
            return redirect(url_for('login'))
        return view(*a, **kw)
    return wrapped


def start_session(user_id):
    session.clear()                     # fresh session = no fixation
    session['uid'] = user_id
    session.permanent = False
    csrf_token()


# very small brute-force guard: 5 wrong passwords per IP+username per 10 minutes
_FAILS = {}
def _throttled(key):
    now = datetime.datetime.utcnow().timestamp()
    _FAILS[key] = [t for t in _FAILS.get(key, []) if now - t < 600]
    return len(_FAILS[key]) >= 5
def _record_fail(key):
    _FAILS.setdefault(key, []).append(datetime.datetime.utcnow().timestamp())


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

def country_options():
    # Pakistan first, then the rest in their usual order
    return ['Pakistan'] + [c for c in COUNTRY_LIST if c != 'Pakistan']


@app.route('/')
def index():
    prefill = {}
    if g.user:
        prefill = {'patient_name': g.user['full_name'], 'gender': g.user['gender'],
                   'age_group': g.user['age_group'], 'country': g.user['country']}
    return render_template('index.html',
                           report_title=REPORT_TITLE,
                           model_accuracy=MODEL_ACCURACY,
                           countries=country_options(),
                           prefill={k: v for k, v in prefill.items() if v})


@app.route('/predict', methods=['POST'])
def predict_route():
    problems = validate_assessment(request.form)
    if problems:
        flash('Please answer all questions marked with * before generating your report.', 'error')
        return redirect(url_for('index'))

    default_name = g.user['full_name'] if g.user else 'Anonymous'
    result = predict(request.form, default_name=default_name)

    report_id = None
    if g.user:                          # logged in -> record is saved in the db file
        report_id = db.save_assessment(g.user['id'], clean_answers(request.form), result)
    return render_template('result.html', result=result,
                           saved=report_id is not None, report_id=report_id)


# ── Accounts ──────────────────────────────────────────────────────────────────
import re
USERNAME_RE = re.compile(r'^[A-Za-z0-9_.-]{3,30}$')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if g.user:
        return redirect(url_for('profile'))
    form = request.form
    if request.method == 'POST':
        full_name = (form.get('full_name') or '').strip()[:80]
        username  = (form.get('username') or '').strip()
        password  = form.get('password') or ''
        gender    = form.get('gender') if form.get('gender') in GENDERS else None
        age_group = form.get('age_group') if form.get('age_group') in AGE_GROUPS else None
        country   = form.get('country') if form.get('country') in COUNTRY_LIST else None
        error = None
        if not full_name:
            error = 'Please enter your name.'
        elif not USERNAME_RE.match(username):
            error = 'Username must be 3-30 characters: letters, numbers, dot, dash or underscore.'
        elif len(password) < 8:
            error = 'Password must be at least 8 characters.'
        elif password != form.get('password2'):
            error = 'The two passwords do not match.'
        if not error:
            uid = db.create_user(username, full_name, generate_password_hash(password),
                                 gender, age_group, country)
            if uid is None:
                error = 'That username is already taken. Please choose another.'
            else:
                start_session(uid)
                flash('Your profile has been created. Your reports will now be saved here.', 'ok')
                return redirect(url_for('index'))
        flash(error, 'error')
    return render_template('register.html', countries=country_options(),
                           genders=GENDERS, age_groups=AGE_GROUPS, form=form)


_DUMMY_HASH = generate_password_hash('not-a-real-password')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if g.user:
        return redirect(url_for('profile'))
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        password = request.form.get('password') or ''
        key = f"{request.remote_addr}|{username.lower()}"
        if _throttled(key):
            flash('Too many failed attempts. Please wait 10 minutes and try again.', 'error')
            return render_template('login.html'), 429
        row = db.get_user_by_username(username)
        ok = check_password_hash(row['password_hash'] if row else _DUMMY_HASH, password)
        if row and ok:
            start_session(row['id'])
            return redirect(url_for('profile'))
        _record_fail(key)
        flash('Incorrect username or password.', 'error')
    return render_template('login.html')


@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


# ── Profile & saved records ───────────────────────────────────────────────────
def _trend_svg(rows):
    """Tiny inline SVG line of overall score over time (oldest -> newest)."""
    pts = [r['overall_score'] for r in reversed(rows)][-20:]
    if len(pts) < 2:
        return ''
    w, h, pad = 320, 70, 8
    step = (w - 2 * pad) / (len(pts) - 1)
    xy = [(pad + i * step, pad + (1 - v / 10) * (h - 2 * pad)) for i, v in enumerate(pts)]
    line = ' '.join(f'{x:.1f},{y:.1f}' for x, y in xy)
    dots = ''.join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#1E3A5F"/>' for x, y in xy)
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
            f'aria-label="Overall score over time (lower is better)">'
            f'<polyline points="{line}" fill="none" stroke="#1E3A5F" stroke-width="2"/>{dots}</svg>')


@app.route('/profile')
@login_required
def profile():
    rows = db.list_assessments(g.user['id'])
    history = [{**dict(r), 'color': RISK_COLORS[r['overall_risk']],
                'bg': RISK_BG[r['overall_risk']]} for r in rows]
    return render_template('profile.html', history=history, trend=_trend_svg(rows),
                           countries=country_options(), genders=GENDERS, age_groups=AGE_GROUPS)


@app.route('/profile/update', methods=['POST'])
@login_required
def profile_update():
    full_name = (request.form.get('full_name') or '').strip()[:80]
    if not full_name:
        flash('Name cannot be empty.', 'error')
        return redirect(url_for('profile'))
    f = request.form
    db.update_user(g.user['id'], full_name,
                   f.get('gender') if f.get('gender') in GENDERS else None,
                   f.get('age_group') if f.get('age_group') in AGE_GROUPS else None,
                   f.get('country') if f.get('country') in COUNTRY_LIST else None)
    flash('Profile updated.', 'ok')
    return redirect(url_for('profile'))


@app.route('/report/<int:report_id>')
@login_required
def view_report(report_id):
    rec = db.get_assessment(g.user['id'], report_id)
    if rec is None:
        abort(404)
    return render_template('result.html', result=rec['result'], saved=True,
                           report_id=report_id, past=True)


@app.route('/report/<int:report_id>/delete', methods=['POST'])
@login_required
def delete_report(report_id):
    db.delete_assessment(g.user['id'], report_id)
    flash('Report deleted.', 'info')
    return redirect(url_for('profile'))


@app.route('/profile/delete', methods=['POST'])
@login_required
def delete_profile():
    if not check_password_hash(g.user['password_hash'], request.form.get('password') or ''):
        flash('Password incorrect - your profile was not deleted.', 'error')
        return redirect(url_for('profile'))
    db.delete_user(g.user['id'])
    session.clear()
    flash('Your profile and all saved reports have been permanently deleted.', 'info')
    return redirect(url_for('index'))


# ── Ask for help -> opens a Gmail message to the help inbox ──────────────────
@app.route('/ask-help')
def ask_help():
    return redirect(help_url())


if __name__ == '__main__':
    app.run(debug=True, port=5000)
