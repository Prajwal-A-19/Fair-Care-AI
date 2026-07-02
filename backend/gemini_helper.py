import google.generativeai as genai
import os, re, time, json
from dotenv import load_dotenv

load_dotenv()
_model = None

def _get_model():
    global _model
    if _model is None:
        key = os.getenv('GEMINI_API_KEY', '').strip()
        if not key:
            raise RuntimeError('GEMINI_API_KEY not set in .env file')
        genai.configure(api_key=key)
        _model = genai.GenerativeModel('gemini-2.0-flash')
    return _model

def _call(prompt: str, retries: int = 3) -> str | None:
    for attempt in range(retries):
        try:
            r    = _get_model().generate_content(prompt)
            text = (r.text or '').strip()
            if len(text) > 20:
                return text
        except Exception as e:
            print(f'[Gemini] Attempt {attempt+1}: {e}')
            if attempt < retries - 1:
                time.sleep(1)
    return None

def _clean(text: str) -> str:
    """Remove markdown symbols Gemini sometimes adds."""
    text = re.sub(r'\*{1,2}', '', text)
    text = re.sub(r'#{1,3}\s*', '', text)
    return text.strip()

def _parse_actions(text: str) -> tuple[str, list[str]]:
    """Split Gemini response into explanation + action list."""
    marker = 'ACTIONS:'
    if marker in text:
        parts = text.split(marker, 1)
        explanation = _clean(parts[0])
        raw_actions = parts[1].strip().split('\n')
        actions = []
        for line in raw_actions:
            line = line.strip()
            if not line:
                continue
            # strip leading "1. " or "- "
            line = re.sub(r'^[\d]+[.)]\s*', '', line)
            line = re.sub(r'^[-•]\s*', '', line)
            cleaned = _clean(line)
            if cleaned:
                actions.append(cleaned)
        return explanation, actions[:3]
    return _clean(text), []

# ═══════════════════════════════════════════════════════════════
#  PATIENT BIAS EXPLANATION
# ═══════════════════════════════════════════════════════════════

def explain_patient(result: dict, language: str = 'English') -> tuple[str, list[str]]:
    m   = result.get('medical', {})
    sym = m.get('symptom', 'presenting symptoms')
    bp  = m.get('bp_systolic', 120)
    pain= m.get('pain_level', 5)
    card= m.get('cardiac_history', 'No')
    diab= m.get('diabetes', 'No')
    gap = result['gender_gap']
    sev = result['bias_severity']
    und = result['underscored_by']

    prompt = f"""You are a senior medical ethics officer reviewing a real-time AI bias alert in an Indian hospital emergency department.

PATIENT PROFILE:
  Name: {result['name']}
  Gender: {result['gender']}
  Age: {result['age']}
  Presenting symptom: {sym}
  Blood pressure: {bp}/80 mmHg
  Pain level: {pain}/10
  Cardiac history: {card}
  Diabetes: {diab}

AI TRIAGE SCORES:
  Current biased hospital AI: {result['biased_score']}% urgency
  FairCare corrected score:   {result['fair_score']}% urgency
  If patient were {result['cf_gender']}: {result['cf_score']}% urgency (counterfactual)
  Gender gap:                 {gap}% — {sev} severity
  Clinical urgency from vitals: {result['clinical_urgency']}%
  Patient underscored by:     {und}%
  Model confidence:           {result.get('model_confidence', 85)}%

Write your response in {language}.

Write exactly this structure:

EXPLANATION:
Two sentences. Sentence 1: What this {gap}% gender gap means for {result['name']} in the emergency room right now — be specific, mention the symptom and the numbers. Sentence 2: Why the gap is unjustifiable — mention that an identical {result['cf_gender']} patient would score {result['cf_score']}% with no medical reason for the difference.

ACTIONS:
1. [Immediate action the doctor should take in the next 5 minutes — specific, clinical]
2. [Diagnostic or escalation step based on {sym} and vitals — specific]
3. [Documentation or system action — mention the AI override and compliance record]

Be direct. Use the patient's name. Make every number count. No fluff."""

    response = _call(prompt)

    if response:
        # parse EXPLANATION: and ACTIONS: sections
        if 'EXPLANATION:' in response:
            parts = response.split('EXPLANATION:', 1)
            rest = parts[1]
            explanation_part, actions = _parse_actions(rest)
        else:
            explanation_part, actions = _parse_actions(response)

        explanation = _clean(explanation_part)
        if len(explanation) < 30:
            explanation = _fallback_patient_explanation(result)
        if not actions:
            actions = _fallback_actions(result)
    else:
        explanation = _fallback_patient_explanation(result)
        actions     = _fallback_actions(result)

    return explanation, actions

def _fallback_patient_explanation(r: dict) -> str:
    m   = r.get('medical', {})
    sym = m.get('symptom', 'their symptoms')
    return (
        f"{r['name']} is presenting with {sym} and a pain level of "
        f"{m.get('pain_level', '—')}/10, yet the hospital AI scores their "
        f"triage urgency at only {r['biased_score']}% — "
        f"{r['underscored_by']}% below what their clinical vitals indicate. "
        f"An identical {r['cf_gender']} patient with the same history would "
        f"score {r['cf_score']}%, a {r['gender_gap']}% difference that has "
        f"no medical justification and constitutes {r['bias_severity'].lower()}"
        f"-severity algorithmic discrimination."
    )

def _fallback_actions(r: dict) -> list[str]:
    m   = r.get('medical', {})
    sym = m.get('symptom', 'presenting symptom')
    cardiac = m.get('cardiac_history') == 'Yes'
    return [
        f"Override triage score to {r['fair_score']}% urgency immediately — "
        f"do not wait for the AI to self-correct",
        f"Order {'12-lead ECG and cardiac enzyme panel' if cardiac or 'chest' in sym.lower() else 'full vitals reassessment and relevant diagnostics'} within 10 minutes",
        f"Document AI override in patient record with FairCare audit reference — "
        f"required for hospital compliance under ICMR AI ethics guidelines",
    ]

# ═══════════════════════════════════════════════════════════════
#  AGGREGATE AUDIT EXPLANATION
# ═══════════════════════════════════════════════════════════════

def explain_aggregate(result: dict, language: str = 'English') -> str:
    mb  = result.get('metrics_before', {})
    ma  = result.get('metrics_after',  {})
    fi  = result.get('feature_importance', {})
    top_feat = max(fi, key=fi.get) if fi else 'Gender'

    prompt = f"""You are a medical ethics AI summarising a hospital AI fairness audit for a senior administrator in India.

AUDIT FINDINGS — {result['total_records']} patient records:

BIAS METRICS (before correction → after correction):
  Demographic Parity gap:  {mb.get('demographic_parity_diff', '?')}% → {ma.get('demographic_parity_diff', '?')}%
  Disparate Impact Ratio:  {mb.get('disparate_impact_ratio', '?')} → {ma.get('disparate_impact_ratio', '?')}
    (below 0.8 = legally actionable discrimination)
  Equalized Odds gap:      {mb.get('equalized_odds_diff', '?')}% → {ma.get('equalized_odds_diff', '?')}%
  Calibration Error:       {mb.get('calibration_error', '?')}% → {ma.get('calibration_error', '?')}%

PATIENT SCORES:
  Female triage score:  {result['female_before']}% → {result['female_after']}%
  Male triage score:    {result['male_before']}% → {result['male_after']}%
  Gender gap:           {result['gap_before']}% → {result['gap_after']}%

TOP BIAS DRIVER: {top_feat} (contributes {fi.get(top_feat, '—')}% of model bias)
Estimated patients affected monthly: {result['patients_affected']}

Write in {language}. Write exactly 3 sentences:
Sentence 1: What the audit found — use specific numbers, mention the disparate impact ratio and whether it crosses the 0.8 legal threshold.
Sentence 2: The dominant cause of bias — which factor ({top_feat}) drives it and why that is a systemic problem in hospital data.
Sentence 3: What the hospital must do immediately — specific, urgent, mention the corrected model improves the gap to {result['gap_after']}%.

Be direct. No bullet points. Sound like an ethics officer, not a chatbot."""

    response = _call(prompt)
    if response and len(_clean(response)) > 40:
        return _clean(response)

    # robust fallback
    di = mb.get('disparate_impact_ratio', '—')
    legal = 'below the 0.8 legal threshold — constituting actionable algorithmic discrimination' \
            if isinstance(di, float) and di < 0.8 else 'within legal range but ethically concerning'
    return (
        f"This audit of {result['total_records']} records reveals female patients "
        f"receive triage scores {result['gap_before']}% lower than male patients, "
        f"with a disparate impact ratio of {di} — {legal}. "
        f"The primary driver is {top_feat}, contributing {fi.get(top_feat, '—')}% of total "
        f"model bias, a direct result of historical underrepresentation in the training data. "
        f"The hospital must deploy the corrected model immediately — it reduces the gender gap "
        f"from {result['gap_before']}% to {result['gap_after']}%, protecting an estimated "
        f"{result['patients_affected']} patients every month."
    )

# ═══════════════════════════════════════════════════════════════
#  FEATURE EXPLANATION  (new — for "why is it biased" section)
# ═══════════════════════════════════════════════════════════════

def explain_features(feature_importance: dict, language: str = 'English') -> str:
    sorted_feats = sorted(feature_importance.items(),
                          key=lambda x: x[1], reverse=True)
    top3 = sorted_feats[:3]
    top3_str = ', '.join(f'{k} ({v}%)' for k, v in top3)

    prompt = f"""You are explaining to a hospital administrator why their AI is biased.

The top 3 factors driving the bias in the hospital's triage AI are:
{top3_str}

Write 1 sentence in {language} explaining in plain language why these factors cause unfair triage decisions.
Do not use technical terms. Focus on the real-world consequence for patients.
Keep it under 40 words."""

    response = _call(prompt)
    if response and len(_clean(response)) > 20:
        return _clean(response)
    top_name, top_val = top3[0]
    return (
        f"The AI gives disproportionate weight to {top_name.lower()} "
        f"({top_val}% contribution), which is correlated with gender in the training data, "
        f"causing the system to systematically undervalue female patients' urgency."
    )


# ═══════════════════════════════════════════════════════
#  DYNAMIC DISEASE INFO — called from app.py (no circular import)
# ═══════════════════════════════════════════════════════
def get_disease_info(symptom: str, age: int, cardiac: bool,
                     diabetic: bool, bp_high: bool, pain: int) -> dict | None:
    """
    Ask Gemini for ICD-10 code, MTS triage level, and treatments
    for ANY symptom. Returns dict or None if Gemini unavailable.
    Called from app.py — never from bias_engine.py.
    """
    risks = []
    if cardiac:  risks.append('cardiac history')
    if diabetic: risks.append('diabetes')
    if bp_high:  risks.append('hypertension BP>140')
    risk_str = ', '.join(risks) if risks else 'none'

    prompt = f"""You are a clinical decision support AI for an Indian hospital emergency department.

Patient: Age {age}, presenting with "{symptom}", pain level {pain}/10, risk factors: {risk_str}.

Return ONLY valid JSON (no markdown, no extra text):
{{
  "icd10": "most specific ICD-10 code",
  "icd10_desc": "full ICD-10 description",
  "mts_level": 2,
  "mts_name": "Very Urgent",
  "mts_colour": "Orange",
  "mts_time": "10 min",
  "base_urgency": 80,
  "disease_category": "cardiac",
  "treatments": ["specific treatment 1","specific treatment 2","specific treatment 3","specific treatment 4","specific treatment 5"],
  "clinical_notes": "one sentence main clinical concern"
}}

MTS rules: Level 1 Red=life threatening 0min. Level 2 Orange=very urgent 10min. Level 3 Yellow=urgent 60min. Level 4 Green=standard 120min. Level 5 Blue=non-urgent 240min.
base_urgency must match mts_level: Red=90-99, Orange=70-89, Yellow=45-69, Green=20-44, Blue=1-19.
treatments must be specific drug names or test names, not vague instructions.
disease_category must be one of: cardiac, neurological, respiratory, infectious, gastrointestinal, trauma, musculoskeletal, other."""

    resp = _call(prompt)
    if resp:
        try:
            clean = re.sub(r'```json|```', '', resp).strip()
            return json.loads(clean)
        except Exception:
            pass
    return None