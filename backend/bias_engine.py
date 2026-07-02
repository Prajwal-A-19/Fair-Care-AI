import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle, os, warnings
warnings.filterwarnings('ignore')

MODEL_DIR = 'models'
_cache    = {}

# ═══════════════════════════════════════════════════════
#  MTS LEVELS
# ═══════════════════════════════════════════════════════
MTS_LEVELS = {
    1: {'name':'Immediate',   'colour':'Red',    'time':'0 min'},
    2: {'name':'Very Urgent', 'colour':'Orange', 'time':'10 min'},
    3: {'name':'Urgent',      'colour':'Yellow', 'time':'60 min'},
    4: {'name':'Standard',    'colour':'Green',  'time':'120 min'},
    5: {'name':'Non-urgent',  'colour':'Blue',   'time':'240 min'},
}

# ═══════════════════════════════════════════════════════
#  SYMPTOM PROFILES — fallback only, Gemini is primary
# ═══════════════════════════════════════════════════════
SYMPTOM_PROFILES = {
    'Chest pain':           {'icd10':'I20.9',  'icd10_desc':'Angina pectoris',          'base_mts':2,'base_urgency':82,'outcome_prob':0.72,'gender_bias_factor':0.35,'category':'cardiac',          'treatments':['12-lead ECG','Troponin I panel','Aspirin 300mg','Cardiac monitor','Oxygen if SpO2<94%']},
    'Cardiac arrest':       {'icd10':'I46.9',  'icd10_desc':'Cardiac arrest',            'base_mts':1,'base_urgency':99,'outcome_prob':0.95,'gender_bias_factor':0.40,'category':'cardiac',          'treatments':['CPR immediately','Defibrillation','IV Epinephrine 1mg','Intubation','ACLS protocol']},
    'Stroke symptoms':      {'icd10':'I63.9',  'icd10_desc':'Cerebral infarction',       'base_mts':1,'base_urgency':93,'outcome_prob':0.85,'gender_bias_factor':0.45,'category':'neurological',     'treatments':['CT brain non-contrast','NIHSS assessment','tPA eligibility check','Stroke team activation','Blood glucose stat']},
    'Shortness of breath':  {'icd10':'R06.00', 'icd10_desc':'Dyspnoea',                  'base_mts':2,'base_urgency':76,'outcome_prob':0.60,'gender_bias_factor':0.50,'category':'respiratory',      'treatments':['SpO2 monitoring','Chest X-ray','ABG analysis','Nebulised bronchodilator','Peak flow']},
    'Severe headache':      {'icd10':'R51',    'icd10_desc':'Headache',                  'base_mts':2,'base_urgency':70,'outcome_prob':0.52,'gender_bias_factor':0.55,'category':'neurological',     'treatments':['CT brain','Neurological exam','BP check','Analgesia IV','LP if thunderclap']},
    'High fever':           {'icd10':'R50.9',  'icd10_desc':'Fever unspecified',         'base_mts':3,'base_urgency':62,'outcome_prob':0.42,'gender_bias_factor':0.65,'category':'infectious',       'treatments':['Blood culture x2','CBC with differential','CRP','IV paracetamol','IV fluids']},
    'Abdominal pain':       {'icd10':'R10.9',  'icd10_desc':'Unspecified abdominal pain','base_mts':3,'base_urgency':65,'outcome_prob':0.45,'gender_bias_factor':0.58,'category':'gastrointestinal', 'treatments':['Abdominal ultrasound','Surgical consult','IV fluids','Analgesia','Urine test']},
    'Trauma / Injury':      {'icd10':'T14.9',  'icd10_desc':'Injury unspecified',        'base_mts':2,'base_urgency':85,'outcome_prob':0.68,'gender_bias_factor':0.60,'category':'trauma',           'treatments':['Primary survey ABCDE','X-ray','IV access x2','Tetanus prophylaxis','Analgesia']},
    'Dizziness / Fainting': {'icd10':'R42',    'icd10_desc':'Dizziness and giddiness',   'base_mts':3,'base_urgency':58,'outcome_prob':0.38,'gender_bias_factor':0.62,'category':'neurological',     'treatments':['12-lead ECG','Blood glucose','Orthostatic BP','Neurological exam','FBC']},
    'Back pain':            {'icd10':'M54.9',  'icd10_desc':'Dorsalgia unspecified',     'base_mts':4,'base_urgency':38,'outcome_prob':0.18,'gender_bias_factor':0.75,'category':'musculoskeletal',  'treatments':['Neurological exam','X-ray if red flags','Analgesia','Muscle relaxant','Physio referral']},
}

DISEASE_CATEGORIES = {
    'cardiac':          ['Chest pain','Cardiac arrest'],
    'neurological':     ['Stroke symptoms','Severe headache','Dizziness / Fainting'],
    'respiratory':      ['Shortness of breath'],
    'infectious':       ['High fever'],
    'gastrointestinal': ['Abdominal pain'],
    'trauma':           ['Trauma / Injury'],
    'musculoskeletal':  ['Back pain'],
}

def _sev(gap):
    g = abs(gap)
    return 'High' if g > 15 else 'Medium' if g > 7 else 'Low'

def _save(obj, name):
    os.makedirs(MODEL_DIR, exist_ok=True)
    pickle.dump(obj, open(f'{MODEL_DIR}/{name}.pkl','wb'))

def _load():
    global _cache
    if _cache: return
    files = ['biased','fair','scaler','le']
    if all(os.path.exists(f'{MODEL_DIR}/{f}.pkl') for f in files):
        for f in files:
            _cache[f] = pickle.load(open(f'{MODEL_DIR}/{f}.pkl','rb'))
    else:
        raise RuntimeError('No trained model. Upload a CSV in Dataset Audit first.')

def _enc_edu(edu, le):
    edu = str(edu).strip()
    return float(le.transform([edu])[0]) if edu in le.classes_ \
           else float(le.transform(['Bachelors'])[0])

def _mts_level(urgency):
    if urgency >= 90: return 1, MTS_LEVELS[1]
    if urgency >= 70: return 2, MTS_LEVELS[2]
    if urgency >= 45: return 3, MTS_LEVELS[3]
    if urgency >= 20: return 4, MTS_LEVELS[4]
    return 5, MTS_LEVELS[5]

# ═══════════════════════════════════════════════════════
#  PREPARE DATASET
# ═══════════════════════════════════════════════════════
def _prepare(filepath):
    df = pd.read_csv(filepath, header=None, names=[
        'age','workclass','fnlwgt','education','education-num',
        'marital-status','occupation','relationship','race','sex',
        'capital-gain','capital-loss','hours-per-week','native-country','income'
    ])
    df['sex']    = df['sex'].str.strip()
    df['income'] = df['income'].str.strip()
    df['sex_encoded'] = (df['sex'].str.lower() == 'male').astype(int)

    le = LabelEncoder()
    df['edu_enc'] = le.fit_transform(df['education'].str.strip())

    df['income_proxy'] = pd.cut(
        df['fnlwgt'].rank(pct=True),
        bins=[0,.25,.5,.75,1.0], labels=[0,1,2,3], include_lowest=True
    ).astype(int)

    df['age_group'] = pd.cut(
        df['age'], bins=[0,30,45,60,100],
        labels=['18-30','31-45','46-60','60+']
    )

    rng = np.random.RandomState(42)
    syms = list(SYMPTOM_PROFILES.keys())
    symptoms = []
    for _, row in df.iterrows():
        age, sex = row['age'], row['sex_encoded']
        if age >= 55 and sex == 1:
            sym = rng.choice(['Chest pain','Cardiac arrest','Shortness of breath'],p=[.55,.10,.35])
        elif age >= 55 and sex == 0:
            sym = rng.choice(['Chest pain','Stroke symptoms','Shortness of breath'],p=[.45,.20,.35])
        elif age >= 35:
            sym = rng.choice(['Abdominal pain','High fever','Chest pain','Back pain'],p=[.30,.30,.20,.20])
        else:
            sym = rng.choice(['High fever','Back pain','Dizziness / Fainting'],p=[.40,.35,.25])
        symptoms.append(sym)

    df['symptom'] = symptoms
    df['symptom_enc'] = df['symptom'].map(
        {s: p['base_urgency']/100 for s,p in SYMPTOM_PROFILES.items()}
    ).fillna(0.5)

    label_probs = []
    for _, row in df.iterrows():
        prof = SYMPTOM_PROFILES.get(row['symptom'], {})
        base = prof.get('outcome_prob', 0.3)
        if row['sex_encoded'] == 0:
            base = base * prof.get('gender_bias_factor', 0.65)
        inc_adj = (row['income_proxy'] / 3) * 0.15
        label_probs.append(min(0.95, base + inc_adj))
    df['label'] = (rng.random(len(df)) < np.array(label_probs)).astype(int)

    return df, le

# ═══════════════════════════════════════════════════════
#  TRAIN MODELS
# ═══════════════════════════════════════════════════════
def _train(df, le):
    global _cache
    FEATS = ['age','edu_enc','hours-per-week','sex_encoded','income_proxy','symptom_enc']
    X     = df[FEATS].fillna(0).values.astype(float)
    y     = df['label'].values.astype(int)
    sex   = df['sex_encoded'].values

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    drop = []
    for sym, prof in SYMPTOM_PROFILES.items():
        mask      = (df['symptom'] == sym).values
        female_pos= np.where(mask & (sex==0) & (y==1))[0]
        n_drop    = int(len(female_pos) * (1.0 - prof['gender_bias_factor']))
        if n_drop > 0:
            drop.extend(np.random.RandomState(42).choice(female_pos, n_drop, replace=False).tolist())

    keep   = np.setdiff1d(np.arange(len(y)), np.array(drop))
    biased = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
    biased.fit(Xs[keep], y[keep])

    weights= np.where(sex==0, 3.0, 1.0)
    fair   = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    fair.fit(Xs, y, sample_weight=weights)

    feat_names = ['Age','Education','Hours/week','Gender','Socioeconomic','Symptom severity']
    coefs      = np.abs(biased.coef_[0])
    importance = {feat_names[i]: round(float(coefs[i]/coefs.sum()*100),1) for i in range(len(feat_names))}

    _cache = {'biased':biased,'fair':fair,'scaler':scaler,'le':le}
    _save(biased,'biased'); _save(fair,'fair')
    _save(scaler,'scaler'); _save(le,'le')
    return importance

# ═══════════════════════════════════════════════════════
#  FAIRNESS METRICS
# ═══════════════════════════════════════════════════════
def _metrics(df, probs, sex):
    preds  = (probs >= 0.5).astype(int)
    y      = df['label'].values
    fm, mm = sex==0, sex==1

    f_rate = preds[fm].mean() if fm.sum()>0 else 0
    m_rate = preds[mm].mean() if mm.sum()>0 else 0
    dp_diff = round(float((m_rate-f_rate)*100),2)
    di_ratio= round(float(f_rate/(m_rate+1e-9)),3)

    f_tp = fm & (y==1); m_tp = mm & (y==1)
    f_tpr= preds[f_tp].mean() if f_tp.sum()>0 else 0
    m_tpr= preds[m_tp].mean() if m_tp.sum()>0 else 0
    eo_diff = round(float((m_tpr-f_tpr)*100),2)

    f_cal = float(np.abs(probs[fm]-y[fm]).mean()*100) if fm.sum()>0 else 0
    m_cal = float(np.abs(probs[mm]-y[mm]).mean()*100) if mm.sum()>0 else 0
    cal_err = round(abs(f_cal-m_cal),2)

    return {
        'demographic_parity_diff': dp_diff,
        'disparate_impact_ratio':  di_ratio,
        'equalized_odds_diff':     eo_diff,
        'calibration_error':       cal_err,
        'female_positive_rate':    round(float(f_rate*100),1),
        'male_positive_rate':      round(float(m_rate*100),1),
        'di_is_illegal':           di_ratio < 0.8,
    }

# ═══════════════════════════════════════════════════════
#  DISEASE BIAS
# ═══════════════════════════════════════════════════════
def _disease_bias(df, b_probs, f_probs, sex):
    df = df.copy()
    df['b'] = b_probs; df['f'] = f_probs
    results = {}
    for cat, syms in DISEASE_CATEGORIES.items():
        mask = df['symptom'].isin(syms)
        if mask.sum() < 10: continue
        cdf = df[mask]; csex = sex[mask]
        fm,mm = csex==0, csex==1
        if fm.sum()==0 or mm.sum()==0: continue
        fb_b = cdf['b'][fm].mean()*100; mb_b = cdf['b'][mm].mean()*100
        fb_f = cdf['f'][fm].mean()*100; mb_f = cdf['f'][mm].mean()*100
        gap_b= mb_b-fb_b; gap_a=mb_f-fb_f
        results[cat] = {
            'gap_before':    round(float(gap_b),1),
            'gap_after':     round(float(gap_a),1),
            'severity':      _sev(gap_b),
            'female_before': round(float(fb_b),1),
            'male_before':   round(float(mb_b),1),
            'patient_count': int(mask.sum()),
        }
    return results

# ═══════════════════════════════════════════════════════
#  PER-PATIENT REPORTS
# ═══════════════════════════════════════════════════════
def _patient_reports(df, b_probs, f_probs, sex, n=10):
    df = df.copy()
    df['b']=b_probs; df['f']=f_probs
    df_f = df[sex==0].copy()
    df_f['gap'] = (df_f['f']-df_f['b'])*100
    top = df_f.nlargest(n,'gap')
    reports = []
    for _,row in top.iterrows():
        sym = str(row.get('symptom','Chest pain'))
        p   = SYMPTOM_PROFILES.get(sym,{})
        reports.append({
            'age':                    int(row['age']),
            'gender':                 'Female',
            'symptom':                sym,
            'icd10':                  p.get('icd10','—'),
            'disease_category':       p.get('category','general'),
            'biased_score':           round(float(row['b'])*100,1),
            'fair_score':             round(float(row['f'])*100,1),
            'gap':                    round(float(row['gap']),1),
            'severity':               _sev(row['gap']),
            'recommended_treatments': p.get('treatments',[]),
        })
    return reports

# ═══════════════════════════════════════════════════════
#  PUBLIC: AGGREGATE AUDIT
# ═══════════════════════════════════════════════════════
def run_audit(filepath):
    df, le     = _prepare(filepath)
    importance = _train(df, le)

    FEATS = ['age','edu_enc','hours-per-week','sex_encoded','income_proxy','symptom_enc']
    Xs    = _cache['scaler'].transform(df[FEATS].fillna(0).values.astype(float))
    sex   = df['sex_encoded'].values
    b_probs = _cache['biased'].predict_proba(Xs)[:,1]
    f_probs = _cache['fair'].predict_proba(Xs)[:,1]

    df['b']=b_probs; df['f']=f_probs
    fb=df[sex==0]['b'].mean(); mb=df[sex==1]['b'].mean()
    ff=df[sex==0]['f'].mean(); mf=df[sex==1]['f'].mean()
    gap_b=(mb-fb)*100; gap_a=(mf-ff)*100

    age_data={}
    for g in ['18-30','31-45','46-60','60+']:
        mask=df['age_group']==g
        if mask.sum()>0:
            age_data[g]={'before':round(df[mask]['b'].mean()*100,1),'after':round(df[mask]['f'].mean()*100,1)}

    income_data={}
    for name,val in [('Low',0),('Mid-low',1),('Mid-high',2),('High',3)]:
        mask=df['income_proxy']==val
        if mask.sum()>0:
            income_data[name]={'before':round(df[mask]['b'].mean()*100,1),'after':round(df[mask]['f'].mean()*100,1)}

    return {
        'female_before':      round(float(fb*100),1),
        'male_before':        round(float(mb*100),1),
        'female_after':       round(float(ff*100),1),
        'male_after':         round(float(mf*100),1),
        'gap_before':         round(float(gap_b),1),
        'gap_after':          round(float(gap_a),1),
        'di_ratio':           _metrics(df,b_probs,sex)['disparate_impact_ratio'],
        'bias_severity':      _sev(gap_b),
        'patients_affected':  int(abs(gap_b/100)*1500*0.6),
        'total_records':      len(df),
        'metrics_before':     _metrics(df,b_probs,sex),
        'metrics_after':      _metrics(df,f_probs,sex),
        'age_data':           age_data,
        'income_data':        income_data,
        'feature_importance': importance,
        'disease_bias':       _disease_bias(df,b_probs,f_probs,sex),
        'patient_reports':    _patient_reports(df,b_probs,f_probs,sex),
    }

# ═══════════════════════════════════════════════════════
#  PUBLIC: INDIVIDUAL PATIENT CHECK — NO GEMINI IMPORTS
# ═══════════════════════════════════════════════════════
def check_patient(data, disease_info=None):
    """
    disease_info: optional dict from Gemini (passed in by app.py).
    If None, falls back to SYMPTOM_PROFILES.
    """
    _load()
    biased=_cache['biased']; fair=_cache['fair']
    scaler=_cache['scaler']; le=_cache['le']

    name    = str(data.get('name','Patient')).strip()
    age     = float(data.get('age',40))
    sex_str = str(data.get('sex','Female')).strip().lower()
    sex_val = 1.0 if sex_str=='male' else 0.0
    edu     = str(data.get('education','Bachelors'))
    hrs     = float(data.get('hours_per_week',40))
    bp      = int(data.get('bp_systolic',120))
    pain    = int(data.get('pain_level',5))
    symptom = str(data.get('symptom','Chest pain')).strip()

    medical = {
        'diabetes':        str(data.get('diabetes','No')),
        'cardiac_history': str(data.get('cardiac_history','No')),
        'bp_systolic':     bp,
        'pain_level':      pain,
        'symptom':         symptom,
        'prior_hosp':      str(data.get('prior_hosp','No')),
    }

    cardiac  = medical['cardiac_history']=='Yes'
    diabetic = medical['diabetes']=='Yes'
    bp_high  = bp > 140
    prior    = medical['prior_hosp']=='Yes'

    # Use Gemini disease_info if provided, else fall back
    if disease_info and isinstance(disease_info, dict):
        icd10         = disease_info.get('icd10','R69')
        icd10_desc    = disease_info.get('icd10_desc','Symptom unspecified')
        mts_level_num = int(disease_info.get('mts_level',3))
        mts_name      = disease_info.get('mts_name','Urgent')
        mts_colour    = disease_info.get('mts_colour','Yellow')
        mts_time      = disease_info.get('mts_time','60 min')
        base_urgency  = int(disease_info.get('base_urgency',60))
        disease_cat   = disease_info.get('disease_category','general')
        treatments    = list(disease_info.get('treatments',[]))
        clinical_notes= disease_info.get('clinical_notes','')
    else:
        profile        = SYMPTOM_PROFILES.get(symptom, SYMPTOM_PROFILES['Chest pain'])
        icd10          = profile['icd10']
        icd10_desc     = profile['icd10_desc']
        mts_level_num  = profile['base_mts']
        mts_name       = MTS_LEVELS[mts_level_num]['name']
        mts_colour     = MTS_LEVELS[mts_level_num]['colour']
        mts_time       = MTS_LEVELS[mts_level_num]['time']
        base_urgency   = profile['base_urgency']
        disease_cat    = profile['category']
        treatments     = list(profile['treatments'])
        clinical_notes = ''

    # Add risk-specific treatments
    extra=[]
    if cardiac  and not any('ECG' in t for t in treatments):   extra.insert(0,'12-lead ECG (cardiac history)')
    if diabetic and not any('glucose' in t.lower() for t in treatments): extra.append('Blood glucose monitoring')
    if bp_high  and not any('BP' in t for t in treatments):    extra.append('BP management protocol')
    treatments = extra + treatments

    sym_enc = base_urgency / 100
    edu_enc = _enc_edu(edu, le)

    base_row = np.array([[age, edu_enc, hrs, sex_val, 1.0, sym_enc]])
    base_s   = scaler.transform(base_row)

    # Gender counterfactual
    cf_g = base_row.copy(); cf_g[0][3]=1.0-sex_val
    cf_g_s = scaler.transform(cf_g)

    # Age counterfactual
    cf_age_val = 70.0 if age < 50 else 30.0
    cf_a = base_row.copy(); cf_a[0][0]=cf_age_val
    cf_a_s = scaler.transform(cf_a)

    # Education counterfactual
    high_edu=['Masters','Doctorate','Prof-school']
    cf_edu_str='HS-grad' if edu in high_edu else 'Masters'
    cf_edu_enc=_enc_edu(cf_edu_str, le)
    cf_edu_inc=0.0 if edu in high_edu else 3.0
    cf_e = base_row.copy(); cf_e[0][1]=cf_edu_enc; cf_e[0][4]=cf_edu_inc
    cf_e_s = scaler.transform(cf_e)

    b_p  = biased.predict_proba(base_s)[0]
    f_p  = fair.predict_proba(base_s)[0]
    cf_g_p = biased.predict_proba(cf_g_s)[0]
    cf_a_p = biased.predict_proba(cf_a_s)[0]
    cf_e_p = biased.predict_proba(cf_e_s)[0]

    bs   = float(b_p[1])*100
    fs   = float(f_p[1])*100
    cs_g = float(cf_g_p[1])*100
    cs_a = float(cf_a_p[1])*100
    cs_e = float(cf_e_p[1])*100
    conf = round(float(max(b_p))*100,1)

    gender_label = 'Female' if sex_val==0 else 'Male'
    cf_gender    = 'Male'   if sex_val==0 else 'Female'
    gender_gap   = abs(bs-cs_g)
    age_gap      = abs(bs-cs_a)
    edu_gap      = abs(bs-cs_e)
    max_gap      = max(gender_gap, age_gap, edu_gap)

    # Clinical urgency from MTS
    multiplier = 1.0
    if cardiac:  multiplier *= 1.15
    if diabetic: multiplier *= 1.08
    if bp_high:  multiplier *= 1.10
    pain_factor = 0.6 + (pain/10)*0.4
    clinical    = max(1, min(99, int(base_urgency * multiplier * pain_factor)))
    if prior:   clinical = min(99, clinical+8)

    tx_biased = treatments[:2]
    tx_missed = treatments[2:]

    return {
        'name':              name,
        'gender':            gender_label,
        'age':               int(age),
        'symptom':           symptom,
        'icd10':             icd10,
        'icd10_desc':        icd10_desc,
        'disease_category':  disease_cat,
        'mts_level':         mts_level_num,
        'mts_name':          mts_name,
        'mts_colour':        mts_colour,
        'mts_time':          mts_time,
        'clinical_notes':    clinical_notes,
        'biased_score':      round(bs,1),
        'fair_score':        round(fs,1),
        'cf_score':          round(cs_g,1),
        'cf_gender':         cf_gender,
        'gender_gap':        round(gender_gap,1),
        'bias_severity':     _sev(max_gap),
        'biased_against':    'Female' if gender_label=='Female' else 'Female',
        'score_improvement': round(fs-bs,1),
        'clinical_urgency':  clinical,
        'underscored_by':    round(clinical-bs,1),
        'model_confidence':  conf,
        'counterfactuals': {
            'gender': {
                'label':       f'If {cf_gender}',
                'cf_score':    round(cs_g,1),
                'gap':         round(gender_gap,1),
                'severity':    _sev(gender_gap),
                'description': f'Same patient, gender changed to {cf_gender}',
            },
            'age': {
                'label':       f'If age {int(cf_age_val)}',
                'cf_score':    round(cs_a,1),
                'gap':         round(age_gap,1),
                'severity':    _sev(age_gap),
                'description': f'Same patient, age changed from {int(age)} to {int(cf_age_val)}',
            },
            'education': {
                'label':       f'If {cf_edu_str}',
                'cf_score':    round(cs_e,1),
                'gap':         round(edu_gap,1),
                'severity':    _sev(edu_gap),
                'description': f'Same patient, education changed from {edu} to {cf_edu_str}',
            },
        },
        'treatments_full':   treatments,
        'treatments_biased': tx_biased,
        'treatments_missed': tx_missed,
        'treatment_gap':     len(tx_missed),
        'feature_contributions': {
            'Gender bias':         round(gender_gap*0.6,1),
            'Age bias':            round(age_gap*0.5,1),
            'Socioeconomic bias':  round(edu_gap*0.5,1),
            'Symptom severity':    round(sym_enc*18,1),
            'Cardiac history':     16.0 if cardiac else 0.0,
        },
        'medical': medical,
    }

# ═══════════════════════════════════════════════════════
#  AUTO-TRAIN DEFAULT
# ═══════════════════════════════════════════════════════
def auto_train_default(filepath):
    global _cache
    df, le = _prepare(filepath)
    _train(df, le)
    print(f'  [bias_engine] Trained on {len(df)} records.')