from flask import Flask, request, jsonify
from flask_cors import CORS
import os, traceback, time, json, uuid
from bias_engine import run_audit, check_patient, auto_train_default
from gemini_helper import (explain_patient, explain_aggregate,
                            explain_features, get_disease_info)

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

# Rate limiting — graceful if not installed
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    limiter = Limiter(app=app, key_func=get_remote_address,
                      default_limits=['1000 per hour'], storage_uri='memory://')
    def rate(r): return limiter.limit(r)
except Exception:
    def rate(r):
        def d(f): return f
        return d

os.makedirs('uploads', exist_ok=True)
os.makedirs('models',  exist_ok=True)
os.makedirs('data',    exist_ok=True)

PATIENTS_FILE  = 'data/patients.json'
VISITS_FILE    = 'data/visits.json'
AUDITS_FILE    = 'data/audits.json'

def _read(path, default=None):
    try:
        if os.path.exists(path):
            with open(path,'r') as f: return json.load(f)
    except: pass
    return default if default is not None else {}

def _write(path, data):
    try:
        with open(path,'w') as f: json.dump(data, f, default=str, indent=2)
    except Exception as e: print(f'[data] {e}')

def _rlist(path):
    d = _read(path, [])
    return d if isinstance(d,list) else []

def ok(data, code=200):
    data.update({'status':'ok','ts':int(time.time())})
    return jsonify(data), code

def err(msg, code=400):
    return jsonify({'status':'error','message':msg,'ts':int(time.time())}), code

def _validate(data):
    if not data: return 'Request body empty'
    if not str(data.get('name','')).strip(): return 'Patient name required'
    try:
        a = float(data.get('age',0))
        if not 0<a<120: return 'Age must be 1–119'
    except: return 'Age must be a number'
    try:
        b = int(data.get('bp_systolic',0))
        if not 50<b<300: return 'BP must be 50–300 mmHg'
    except: return 'BP must be a number'
    try:
        p = int(data.get('pain_level',5))
        if not 1<=p<=10: return 'Pain level must be 1–10'
    except: return 'Pain level must be a number'
    return None

# ── STARTUP ───────────────────────────────────────────────────────
def _seed_demo_patients():
    patients = _read(PATIENTS_FILE, {})
    if patients:
        return  # already seeded
    demo = [
        {'pid':'FC-DEMO0001','name':'Priya Sharma','age':42,'sex':'Female',
         'symptom':'Chest pain','pain_level':7,'cardiac':True,'diabetic':False,'bp_high':True,
         'education':'Bachelors','hours_per_week':40},
        {'pid':'FC-DEMO0002','name':'Ravi Kumar','age':45,'sex':'Male',
         'symptom':'Chest pain','pain_level':7,'cardiac':True,'diabetic':False,'bp_high':True,
         'education':'Bachelors','hours_per_week':40},
        {'pid':'FC-DEMO0003','name':'Anita Desai','age':67,'sex':'Female',
         'symptom':'Shortness of breath','pain_level':6,'cardiac':False,'diabetic':True,'bp_high':True,
         'education':'HS-grad','hours_per_week':35},
        {'pid':'FC-DEMO0004','name':'Suresh Patel','age':55,'sex':'Male',
         'symptom':'Abdominal pain','pain_level':5,'cardiac':False,'diabetic':True,'bp_high':False,
         'education':'Some-college','hours_per_week':45},
        {'pid':'FC-DEMO0005','name':'Meera Nair','age':34,'sex':'Female',
         'symptom':'High fever','pain_level':4,'cardiac':False,'diabetic':False,'bp_high':False,
         'education':'Masters','hours_per_week':38},
    ]
    for d in demo:
        pid = d['pid']
        patients[pid] = {
            'patient_id':   pid,
            'name':         d['name'],
            'age':          d['age'],
            'sex':          d['sex'],
            'symptom':      d['symptom'],
            'pain_level':   d['pain_level'],
            'cardiac':      d['cardiac'],
            'diabetic':     d['diabetic'],
            'bp_high':      d['bp_high'],
            'education':    d['education'],
            'hours_per_week': d['hours_per_week'],
            'registered_at': '2026-01-01 00:00',
        }
    _write(PATIENTS_FILE, patients)
    print(f'  [FairCare] Seeded {len(demo)} demo patients.')

def startup_check():
    if not os.path.exists('models/biased.pkl'):
        print('\n  [FairCare] Generating training data...')
        try:
            _gen_and_train()
            print('  [FairCare] Ready.\n')
        except Exception as e:
            print(f'  [FairCare] Startup error: {e}\n')
    else:
        print('\n  [FairCare] Models found. Ready.\n')
    _seed_demo_patients()

def _gen_and_train():
    import pandas as pd, numpy as np
    from bias_engine import SYMPTOM_PROFILES
    rng = np.random.RandomState(42)
    n   = 6000
    sex = rng.choice(['Male','Female'], n, p=[0.52,0.48])
    age = rng.normal(45,15,n).clip(18,88).astype(int)
    edu = rng.choice(['Bachelors','HS-grad','Some-college','Masters','Assoc-acdm','Doctorate','Prof-school'],
                     n, p=[0.30,0.28,0.20,0.10,0.06,0.03,0.03])
    hrs    = rng.normal(38,12,n).clip(5,80).astype(int)
    fnlwgt = rng.randint(50000,400000,n)

    syms = []
    for i in range(n):
        a, s = age[i], sex[i]
        if a>=55 and s=='Male':
            sym = rng.choice(['Chest pain','Cardiac arrest','Shortness of breath'],p=[.55,.10,.35])
        elif a>=55 and s=='Female':
            sym = rng.choice(['Chest pain','Stroke symptoms','Shortness of breath'],p=[.45,.20,.35])
        elif a>=35:
            sym = rng.choice(['Abdominal pain','High fever','Chest pain','Back pain'],p=[.30,.30,.20,.20])
        else:
            sym = rng.choice(['High fever','Back pain','Dizziness / Fainting'],p=[.40,.35,.25])
        syms.append(sym)

    labels = []
    for i in range(n):
        prof = SYMPTOM_PROFILES.get(syms[i],{})
        base = prof.get('outcome_prob',0.3)
        if sex[i]=='Female': base *= prof.get('gender_bias_factor',0.65)
        labels.append(1 if rng.random()<base else 0)

    df = pd.DataFrame({
        'age':age,'workclass':'Private','fnlwgt':fnlwgt,
        'education':edu,'education-num':rng.randint(8,17,n),
        'marital-status':'Married','occupation':'Prof-specialty',
        'relationship':'Husband','race':'White','sex':sex,
        'capital-gain':0,'capital-loss':0,'hours-per-week':hrs,
        'native-country':'India',
        'income':['>50K' if l else '<=50K' for l in labels]
    })
    path='uploads/hospital_data.csv'
    df.to_csv(path,index=False,header=False)
    auto_train_default(path)

# ═══════════════════════════════════════════════════════
#  PATIENT MANAGEMENT
# ═══════════════════════════════════════════════════════

@app.route('/patients', methods=['GET'])
def list_patients():
    patients = _read(PATIENTS_FILE,{})
    return ok({'patients': sorted(patients.values(), key=lambda p:p.get('registered_at',''), reverse=True), 'count':len(patients)})

@app.route('/patients/register', methods=['POST'])
def register_patient():
    try:
        data = request.get_json(force=True,silent=True) or {}
        name = str(data.get('name','')).strip()
        if not name: return err('Name required',400)
        pid = 'FC-'+str(uuid.uuid4())[:8].upper()
        patient = {
            'patient_id':       pid,
            'name':             name,
            'age':              data.get('age',''),
            'phone':            str(data.get('phone','')).strip(),
            'blood_group':      str(data.get('blood_group','O+')).strip(),
            'email':            str(data.get('email','')).strip(),
            'allergies':        str(data.get('allergies','')).strip(),
            'emergency_contact':str(data.get('emergency_contact','')).strip(),
            'known_conditions': data.get('known_conditions',[]),
            'registered_at':    time.strftime('%Y-%m-%d %H:%M'),
        }
        patients = _read(PATIENTS_FILE,{})
        patients[pid] = patient
        _write(PATIENTS_FILE, patients)
        return ok({'patient':patient,'patient_id':pid})
    except Exception:
        traceback.print_exc()
        return err('Registration failed',500)

@app.route('/patients/search', methods=['GET'])
def search_patients():
    q = str(request.args.get('q','')).strip().lower()
    if not q: return err('Query required',400)
    patients = _read(PATIENTS_FILE,{})
    results  = [p for p in patients.values()
                if q in p.get('name','').lower()
                or q in p.get('patient_id','').lower()
                or q in p.get('phone','').lower()]
    return ok({'patients':results,'count':len(results)})

@app.route('/patients/<pid>', methods=['GET'])
def get_patient(pid):
    patients = _read(PATIENTS_FILE,{})
    patient  = patients.get(pid)
    if not patient: return err(f'Patient {pid} not found',404)
    visits   = _rlist(VISITS_FILE)
    pt_visits= sorted([v for v in visits if v.get('patient_id')==pid],
                      key=lambda v:v.get('visited_at',''), reverse=True)
    return ok({'patient':patient,'visits':pt_visits,'visit_count':len(pt_visits),
               'last_visit':pt_visits[0].get('visited_at') if pt_visits else None})

@app.route('/patients/<pid>/visit', methods=['POST'])
def add_visit(pid):
    try:
        patients = _read(PATIENTS_FILE,{})
        if pid not in patients: return err(f'Patient {pid} not found',404)
        data  = request.get_json(force=True,silent=True) or {}
        visit = {
            'visit_id':      'V-'+str(uuid.uuid4())[:8].upper(),
            'patient_id':    pid,
            'patient_name':  patients[pid].get('name',''),
            'visited_at':    time.strftime('%Y-%m-%d %H:%M'),
            'doctor_name':   str(data.get('doctor_name','Dr. Unknown')),
            'hospital_name': str(data.get('hospital_name','General Hospital')),
            'symptom':       str(data.get('symptom','')),
            'diagnosis':     str(data.get('diagnosis','')),
            'prescription':  data.get('prescription',[]),
            'notes':         str(data.get('notes','')),
            'follow_up':     str(data.get('follow_up','')),
            'bias_check':    data.get('bias_check',None),
        }
        visits = _rlist(VISITS_FILE)
        visits.insert(0,visit)
        _write(VISITS_FILE, visits[:500])
        return ok({'visit':visit})
    except Exception:
        traceback.print_exc()
        return err('Failed to save visit',500)

# ═══════════════════════════════════════════════════════
#  CORE ROUTES
# ═══════════════════════════════════════════════════════

@app.route('/health')
def health():
    patients = _read(PATIENTS_FILE,{})
    return ok({
        'service':          'FairCare AI Platform v4.0',
        'model_trained':    os.path.exists('models/biased.pkl'),
        'total_patients':   len(patients),
        'total_visits':     len(_rlist(VISITS_FILE)),
        'total_audits':     len(_rlist(AUDITS_FILE)),
        'challenge':        'Google Solution Challenge 2026',
    })

@app.route('/model-status')
def model_status():
    trained  = os.path.exists('models/biased.pkl')
    audits   = _rlist(AUDITS_FILE)
    return ok({
        'ready':            trained,
        'model_type':       'hospital-specific' if (trained and len(audits)>0) else 'general',
        'audit_count':      len(audits),
        'message':          'Using your hospital model' if (trained and len(audits)>0)
                            else 'Using general model — run Dataset Audit for accuracy',
    })

@app.route('/hospital/stats')
def hospital_stats():
    patients = _read(PATIENTS_FILE,{})
    visits   = _rlist(VISITS_FILE)
    audits   = _rlist(AUDITS_FILE)
    checks   = [v['bias_check'] for v in visits if v.get('bias_check')]
    high_bias= sum(1 for b in checks if b.get('bias_severity')=='High')
    avg_gap  = (sum(b.get('gender_gap',0) for b in checks)/len(checks) if checks else 0)
    return ok({
        'total_patients':    len(patients),
        'total_visits':      len(visits),
        'bias_checks_done':  len(checks),
        'high_bias_cases':   high_bias,
        'avg_gender_gap':    round(avg_gap,1),
        'latest_audit':      audits[0] if audits else None,
        'audit_count':       len(audits),
        'patients_protected': int(avg_gap/100*1500*0.6) if avg_gap else 0,
    })

@app.route('/audit-history')
def audit_history():
    return ok({'audits':_rlist(AUDITS_FILE)[:10]})

@app.route('/metrics')
def metrics_info():
    return ok({'metrics':[
        {'name':'Demographic Parity','ideal':'0%',  'legal':'No fixed threshold',         'desc':'Gap in positive outcome rates between genders'},
        {'name':'Disparate Impact',  'ideal':'≥0.8','legal':'<0.8=illegal (ICMR/EEOC)',   'desc':'Ratio of positive rates: female ÷ male'},
        {'name':'Equalized Odds',    'ideal':'0%',  'legal':'>5% clinically significant',  'desc':'Gap in True Positive Rates'},
        {'name':'Calibration Error', 'ideal':'0%',  'legal':'Any gap reduces reliability', 'desc':'Difference in prediction accuracy between groups'},
    ]})

@app.route('/check-patient', methods=['POST'])
@rate('60 per minute')
def check_patient_route():
    try:
        data = request.get_json(force=True,silent=True) or {}
        e = _validate(data)
        if e: return err(e,400)

        lang    = str(data.get('language','English')).strip()
        symptom = str(data.get('symptom','Chest pain')).strip()
        age     = int(data.get('age',40))
        cardiac = data.get('cardiac_history','No')=='Yes'
        diabetic= data.get('diabetes','No')=='Yes'
        bp_high = int(data.get('bp_systolic',120)) > 140
        pain    = int(data.get('pain_level',5))

        # Get dynamic disease info from Gemini (in app.py — no circular import)
        disease_info = get_disease_info(symptom, age, cardiac, diabetic, bp_high, pain)

        # Run ML bias check, passing disease_info
        result = check_patient(data, disease_info=disease_info)

        # Get Gemini explanation
        expl, actions = explain_patient(result, language=lang)
        result['explanation']         = expl
        result['actions']             = actions
        result['feature_explanation'] = explain_features(
            result.get('feature_contributions',{}), language=lang
        )
        return ok(result)

    except RuntimeError as e:
        return err(str(e),503)
    except Exception:
        traceback.print_exc()
        return err('Analysis error',500)

@app.route('/upload-dataset', methods=['POST'])
@rate('10 per hour')
def upload_dataset():
    try:
        if 'file' not in request.files: return err('No file uploaded',400)
        f = request.files['file']
        if not f.filename.lower().endswith('.csv'): return err('Only CSV files',400)
        path='uploads/hospital_data.csv'
        f.save(path)
        if os.path.getsize(path)<500: return err('File too small',400)
        lang   = request.form.get('language','English')
        result = run_audit(path)
        result['explanation']         = explain_aggregate(result,language=lang)
        result['feature_explanation'] = explain_features(result.get('feature_importance',{}),language=lang)
        audits = _rlist(AUDITS_FILE)
        audits.insert(0,{
            'date':              time.strftime('%Y-%m-%d %H:%M'),
            'gap_before':        result.get('gap_before'),
            'gap_after':         result.get('gap_after'),
            'bias_severity':     result.get('bias_severity'),
            'total_records':     result.get('total_records'),
            'patients_affected': result.get('patients_affected'),
            'di_ratio':          result.get('di_ratio'),
        })
        _write(AUDITS_FILE, audits[:20])
        return ok(result)
    except Exception:
        traceback.print_exc()
        return err('Audit failed',500)

# Run startup check whether launched via gunicorn or python directly
startup_check()

if __name__=='__main__':
    app.run(debug=False, port=5000, host='0.0.0.0')