import os
import streamlit as st
import requests
import json
import re
from typing import List, Dict, Any
import pytesseract
from PIL import Image

"""
Streamlit demo: Prescription info extraction + simple interaction checking
Uses Hugging Face Inference API with an IBM Granite instruct model to generate structured output.

Usage:
    1) Set HF_API_TOKEN env var: export HF_API_TOKEN="hf_xxx..."
    2) streamlit run app.py
"""


# Optional OCR
try:
        OCR_AVAILABLE = True
except Exception:
        OCR_AVAILABLE = False

st.set_page_config(page_title="AI Prescription Verifier (IBM Granite)", layout="wide")

st.title("AI Medical Prescription Verifier — (Hugging Face + IBM Generator)")

st.markdown(
        """
This demo uses a Hugging Face IBM Granite *instruct* model via the Hugging Face Inference API
to extract drug names, dosages, frequencies, and durations from free-text or images. 
**This is a prototype — do not use for clinical decisions.**
"""
)

# -------- User inputs ----------
col1, col2 = st.columns([2,1])

with col1:
        input_mode = st.selectbox("Input type", ["Paste text", "Upload image (OCR)"])
        if input_mode == "Paste text":
                prescription_text = st.text_area("Paste the prescription / clinical note here", height=240)
        else:
                uploaded = st.file_uploader("Upload prescription image (png/jpg/pdf)", type=['png','jpg','jpeg'])
                prescription_text = ""
                if uploaded is not None:
                        if OCR_AVAILABLE:
                                try:
                                        img = Image.open(uploaded)
                                        prescription_text = pytesseract.image_to_string(img)
                                        st.text_area("OCR result (editable)", value=prescription_text, height=240)
                                except Exception as e:
                                        st.error(f"OCR failed: {e}")
                        else:
                                st.warning("pytesseract not installed or Tesseract not available. Install to enable OCR.")
                                st.text("Upload ignored until OCR available.")

        age = st.number_input("Patient age (years, optional)", value=30, min_value=0, max_value=130)
        weight = st.number_input("Patient weight (kg, optional)", value=70.0, min_value=0.0)

with col2:
        st.markdown("### Backend model settings")
        hf_model = st.text_input("Hugging Face model id", value="ibm-granite/granite-3.1-8b-instruct")
        hf_token = st.text_input("Hugging Face API token (or set HF_API_TOKEN env var)", type="password")
        if not hf_token:
                hf_token = os.environ.get("HF_API_TOKEN", None)

        max_tokens = st.slider("Max tokens (generation)", 50, 2048, 400)
        temperature = st.slider("Temperature", 0.0, 1.5, 0.2)

st.markdown("---")

# --------- Helper functions ----------
HF_API_URL_BASE = "https://api-inference.huggingface.co/models/"

def generate_with_hf(model_id: str, prompt: str, token: str, max_tokens: int=400, temperature: float=0.2) -> str:
        """
        Call Hugging Face Inference API (text generation endpoint).
        Returns the text output (string). Raises exceptions on errors.
        """
        if not token:
                raise ValueError("Hugging Face API token missing. Set HF_API_TOKEN environment variable or provide it in the UI.")
        url = HF_API_URL_BASE + model_id
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {
                "inputs": prompt,
                "parameters": {
                        "max_new_tokens": max_tokens,
                        "temperature": temperature,
                        "return_full_text": False,
                },
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 200:
                raise RuntimeError(f"Hugging Face API Error {resp.status_code}: {resp.text}")
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
                if isinstance(data[0], dict) and 'generated_text' in data[0]:
                        return data[0]['generated_text']
                else:
                        return "".join([d.get('generated_text', '') if isinstance(d, dict) else str(d) for d in data])
        elif isinstance(data, dict) and 'generated_text' in data:
                return data['generated_text']
        elif isinstance(data, str):
                return data
        else:
                return json.dumps(data)

def prompt_for_extraction(text: str) -> str:
        """
        Create a prompt instructing the model to extract structured drug data.
        The model should return a JSON array of objects with fields:
            - name, dose, frequency, route (optional), duration (optional), notes
        """
        p = f"""
You are a clinical assistant. Extract medication details from the following prescription text.
Return ONLY valid JSON. The JSON must be an array of medication objects. Each object must have these fields:
    - name (string): medication name (brand or generic)
    - dose (string or null): dose as written
    - frequency (string or null): dosing frequency
    - duration (string or null): duration if present
    - route (string or null): oral, iv, topical, etc. if present
    - notes (string or null): any other notes like 'PRN', 'once daily'.

If a field is not present, use null.

Prescription text:
\"\"\"
{text}
\"\"\"

Example output format:
[
    {{"name":"Warfarin","dose":"5 mg","frequency":"once daily","duration":null,"route":"oral","notes":null}},
    {{"name":"Aspirin","dose":"75 mg","frequency":"once daily","duration":null,"route":"oral","notes":null}}
]

Now extract and output ONLY the JSON.
"""
        return p

def extract_json_from_text(text: str) -> Any:
        """
        Try to find a JSON block in the provided text and parse it.
        """
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
                candidate = text[start:end+1]
                try:
                        return json.loads(candidate)
                except Exception:
                        cleaned = re.sub(r",\s*\\]", "]", candidate)
                        cleaned = re.sub(r",\s*\\}", "}", cleaned)
                        try:
                                return json.loads(cleaned)
                        except Exception:
                                return None
        try:
                return json.loads(text)
        except Exception:
                return None

LOCAL_INTERACTIONS = {
        ("warfarin","aspirin"): {"severity":"high","description":"Increased bleeding risk; monitor INR and bleeding."},
        ("ibuprofen","lithium"): {"severity":"moderate","description":"NSAIDs may increase lithium levels."},
}
ALTERNATIVES = {
        "ibuprofen": ["naproxen", "acetaminophen"],
        "aspirin": ["acetaminophen", "clopidogrel"],
}

def normalize(name: str) -> str:
        return name.strip().lower() if name else ""

def check_interactions(meds: List[Dict]) -> List[Dict]:
        flagged = []
        names = [normalize(m.get("name","")) for m in meds]
        for i in range(len(names)):
                for j in range(i+1, len(names)):
                        pair = (names[i], names[j])
                        if pair in LOCAL_INTERACTIONS:
                                info = LOCAL_INTERACTIONS[pair]
                                flagged.append({"pair": [names[i], names[j]], **info})
                        elif (pair[1], pair[0]) in LOCAL_INTERACTIONS:
                                info = LOCAL_INTERACTIONS[(pair[1], pair[0])]
                                flagged.append({"pair": [names[i], names[j]], **info})
        return flagged

def recommend_simple_dosage(meds: List[Dict], age:int=None, weight:float=None) -> Dict:
        recs = {}
        for m in meds:
                n = normalize(m.get("name",""))
                if n in ("paracetamol","acetaminophen"):
                        if age is None or age >= 12:
                                recs[m.get("name")] = "Adult: 500 mg every 4-6 hours (max 4 g/day)."
                        else:
                                recs[m.get("name")] = "Pediatric: 10-15 mg/kg every 4-6 hours."
                elif n == "ibuprofen":
                        recs[m.get("name")] = "200-400 mg every 4-6 hours; max 1200 mg/day OTC (adult guidance)."
                else:
                        recs[m.get("name")] = "No local dosage guidance available. Consult reference."
        return recs

def suggest_alternatives(meds: List[Dict]) -> Dict:
        out = {}
        for m in meds:
                n = normalize(m.get("name",""))
                if n in ALTERNATIVES:
                        out[m.get("name")] = ALTERNATIVES[n]
        return out

# --------- Main operation ----------
if st.button("Analyze with IBM Granite (Hugging Face)"):
        text = prescription_text.strip()
        if not text:
                st.warning("No prescription text provided.")
        else:
                with st.spinner("Generating extraction prompt and calling model..."):
                        try:
                                prompt = prompt_for_extraction(text)
                                raw_out = generate_with_hf(hf_model, prompt, hf_token, max_tokens=max_tokens, temperature=temperature)
                        except Exception as e:
                                st.error(f"Model call failed: {e}")
                                st.stop()

                st.subheader("Raw model output")
                st.code(raw_out[:10000])

                meds = extract_json_from_text(raw_out)
                if meds is None:
                        st.warning("Model did not return parseable JSON. Attempting to extract candidate drug tokens heuristically.")
                        tokens = re.findall(r"[A-Za-z][A-Za-z0-9\\-\\/]{2,}", text)
                        known_drugs = ["paracetamol","acetaminophen","ibuprofen","warfarin","aspirin","amoxicillin","lithium"]
                        found = []
                        for t in tokens:
                                if t.lower() in known_drugs and t.lower() not in [f.lower() for f in found]:
                                        found.append(t)
                        meds = [{"name": d, "dose": None, "frequency": None, "duration": None, "route": None, "notes": None} for d in found]

                st.subheader("Extracted medications (structured)")
                st.write(meds)

                st.subheader("Simple interaction check (toy rules)")
                interactions = check_interactions(meds)
                if interactions:
                        st.warning("Possible interactions found:")
                        st.json(interactions)
                else:
                        st.success("No interactions flagged in local toy database.")

                st.subheader("Dosage recommendations (toy)")
                dosages = recommend_simple_dosage(meds, age=age, weight=weight)
                st.json(dosages)

                st.subheader("Alternative suggestions (toy)")
                alts = suggest_alternatives(meds)
                st.json(alts)

st.markdown("---")
st.markdown(
        "### Important notes\n"
        "- This app uses a general-purpose LLM (IBM Granite) to *extract* information from text. It's a heuristic approach — outputs must be validated by clinicians.\n"
        "- The interaction/dosage rules are toy examples. For production use integrate RxNorm/DrugBank/OpenFDA or IBM Watson drug databases and perform clinical validation.\n"
        "- Running large models locally requires significant GPU memory. Using the Hugging Face Inference API is the simplest route for demos.\n"
)

st.markdown("Model examples on Hugging Face: `ibm-granite/granite-3.1-8b-instruct`, `ibm-granite/granite-3.0-8b-base`. See IBM's Granite model family on Hugging Face for options.")