# UK Employment Law Change Detector

**Automated monitoring of UK employment legislation for material compliance changes affecting SMEs.**

This project connects to the official [legislation.gov.uk](https://www.legislation.gov.uk/) REST API, fetches the current Crown Legislation Markup Language (CLML) XML for four key employment statutes, diffs it against a locally cached baseline, and uses a fine-tuned BERT classifier to separate material compliance changes (new duties, penalty increases, scope extensions) from minor editorial amendments (renumbering, cross-reference fixes, punctuation). The result is a structured JSON alert — complete with domain, effective date, penalty threshold, affected business types, and obligation type — that an SME compliance team can act on immediately, without needing to parse raw legislation.

---

## Architecture

```
+-----------------------------------------------------------------------+
|                        POST /analyse (FastAPI)                        |
+-----------------------------------------------------------------------+
         |                        |                        |
         v                        v                        v
+-------------------+  +---------------------+  +----------------------+
|    LAYER 1        |  |     LAYER 2         |  |      LAYER 3         |
|  Data Ingestion   |  |  ML Classification  |  |  Structured Output   |
+-------------------+  +---------------------+  +----------------------+
|                   |  |                     |  |                      |
| scraper.py        |  | data_builder.py     |  | domain_classifier.py |
|  Fetch CLML XML   |  |  Synthetic training |  |  Act -> domain map   |
|  from legislation |  |  data (100 templates)|  |                      |
|  .gov.uk REST API |  |                     |  | penalty_extractor.py |
|                   |  | classifier.py       |  |  Regex-based penalty |
| diff_engine.py    |  |  BERT fine-tuning   |  |  threshold extraction|
|  Section-level    |  |  (bert-base-uncased)|  |                      |
|  XML diffing      |  |                     |  | output_builder.py    |
|                   |  | scorer.py           |  |  5-field structured  |
| preprocessor.py   |  |  Confidence scoring |  |  JSON records         |
|  Text cleanup for |  |  (threshold: 0.75)  |  |                      |
|  BERT tokenisation|  |                     |  | api.py               |
|                   |  |                     |  |  FastAPI endpoints    |
+-------------------+  +---------------------+  +----------------------+
         |                        |                        |
         v                        v                        v
+-----------------------------------------------------------------------+
|  data/raw/          models/best_model/        outputs/                |
|  Cached XML         Trained BERT checkpoint   JSON alerts & logs      |
+-----------------------------------------------------------------------+
```

### Pipeline Flow

1. **Scrape** — Fetch current CLML XML from legislation.gov.uk
2. **Diff** — Compare against cached baseline (section-level)
3. **Preprocess** — Strip XML, expand abbreviations, normalise for BERT
4. **Classify** — Score each changed section with fine-tuned BERT (material vs minor)
5. **Output** — Build structured JSON with domain, date, penalty, business types, obligation type

---

## Monitored Legislation

| Act | Path | Domain |
|-----|------|--------|
| Employment Rights Act 1996 | `ukpga/1996/18` | employment |
| National Minimum Wage Act 1998 | `ukpga/1998/31` | wage |
| Equality Act 2010 | `ukpga/2010/15` | equality |
| Working Time Regulations 1998 | `uksi/1998/1833` | working_time |

---

## Installation (Windows, Python 3.12)

### Prerequisites

- **Python 3.12** — download from [python.org](https://www.python.org/downloads/)
- **Git** — download from [git-scm.com](https://git-scm.com/downloads)
- **pip** — included with Python 3.12

### Steps

```bash
# Clone the repository
git clone https://github.com/your-username/uk-employment-law-change-detector.git
cd uk-employment-law-change-detector

# Create a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** PyTorch installation may require a platform-specific command. If `pip install torch` fails, visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for the correct install command for your system.

### Verify installation

```bash
pytest tests/
```

All 293 tests should pass (1 skipped — the slow full-training integration test).

---

## Quickstart

### 1. Start the API server

```bash
uvicorn src.api:app --reload --port 8000
```

### 2. Check service health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "service": "uk-employment-law-change-detector",
  "model_available": false
}
```

> `model_available` will be `false` until you train the BERT classifier (see below).

### 3. Train the classifier

```python
from src.data_builder import build_training_dataset
from src.classifier import train

# Generate synthetic training data (200 examples, balanced classes)
build_training_dataset(n_per_class=100, seed=42)

# Fine-tune BERT (saves best model to models/best_model/)
train("data/labelled/training_data.csv")
```

### 4. Analyse legislation for changes

```bash
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d "{\"url\": \"https://www.legislation.gov.uk/ukpga/1996/18\", \"min_confidence\": 0.0}"
```

**First call** saves the current XML as a baseline and returns:

```json
{
  "act_name": "employment_rights_act_1996",
  "pipeline_status": "baseline_established",
  "processed_at": "2026-04-13T10:30:00+00:00",
  "total_material_changes": 0,
  "changes": []
}
```

**Subsequent calls** diff against the baseline. If material changes are detected:

```json
{
  "act_name": "employment_rights_act_1996",
  "pipeline_status": "changes_found",
  "processed_at": "2026-04-13T14:22:00+00:00",
  "total_material_changes": 2,
  "changes": [...]
}
```

---

## Example JSON Output

A single material change record with all five required fields populated:

```json
{
  "section_id": "s47B",
  "change_type": "modified_section",
  "text_excerpt": "The maximum compensatory award for unfair dismissal has been increased from £25,000 to £50,000 per employee...",
  "predicted_label": 1,
  "confidence": 0.94,
  "needs_review": false,
  "domain": "employment",
  "effective_date": "1 April 2026",
  "penalty_threshold": "£50,000 per employee",
  "affected_business_types": ["all employers"],
  "obligation_type": "changed_penalty"
}
```

### The Five Structured Fields

| Field | Description | Example Values |
|-------|-------------|----------------|
| `domain` | Compliance domain derived from the act | `employment`, `wage`, `equality`, `working_time` |
| `effective_date` | Extracted from legislative text, or `null` | `"1 April 2026"`, `"2026-04-01"`, `null` |
| `penalty_threshold` | Monetary penalty amount with qualifiers | `"£20,000"`, `"£500 per day"`, `"unlimited"`, `null` |
| `affected_business_types` | List of affected business categories | `["all employers"]`, `["SMEs"]`, `["public sector employers"]` |
| `obligation_type` | Classification of the legal change | `new_duty`, `changed_duty`, `removed_duty`, `changed_penalty`, `extended_scope` |

---

## Project Structure

```
uk-employment-law-change-detector/
|
|-- src/
|   |-- __init__.py                 # Package marker
|   |-- scraper.py                  # legislation.gov.uk REST API client
|   |-- diff_engine.py              # CLML XML section-level diffing
|   |-- preprocessor.py             # Text cleanup for BERT tokenisation
|   |-- data_builder.py             # Synthetic training data generator
|   |-- classifier.py               # BERT fine-tuning pipeline
|   |-- scorer.py                   # Confidence scoring and inference
|   |-- domain_classifier.py        # Act/URL/text to domain mapping
|   |-- penalty_extractor.py        # Regex-based penalty extraction
|   |-- output_builder.py           # Structured JSON output builder
|   |-- api.py                      # FastAPI application
|
|-- tests/
|   |-- test_scraper.py             # 14 tests
|   |-- test_diff_engine.py         # 25 tests
|   |-- test_preprocessor.py        # 25 tests
|   |-- test_data_builder.py        # 35 tests
|   |-- test_classifier.py          # 27 tests (1 skipped — slow)
|   |-- test_scorer.py              # 28 tests
|   |-- test_domain_classifier.py   # 22 tests
|   |-- test_penalty_extractor.py   # 24 tests
|   |-- test_output_builder.py      # 35 tests
|   |-- test_api.py                 # 19 tests
|
|-- data/
|   |-- raw/                        # Cached legislation XML baselines
|   |-- diffs/                      # Computed XML diffs (JSON)
|   |-- labelled/                   # Generated training datasets (CSV)
|
|-- models/
|   |-- best_model/                 # Trained BERT checkpoint
|
|-- outputs/                        # JSON alerts and training logs
|
|-- requirements.txt                # Python dependencies
|-- pytest.ini                      # Test runner configuration
|-- CHANGELOG.md                    # Version history
|-- README.md                       # This file
```

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.12 |
| **Data source** | legislation.gov.uk REST API (CLML XML) |
| **XML parsing** | lxml |
| **Text preprocessing** | BeautifulSoup4, regex |
| **ML framework** | PyTorch |
| **Pretrained model** | `bert-base-uncased` (HuggingFace Transformers) |
| **Training** | AdamW optimiser, linear warmup, gradient clipping |
| **API framework** | FastAPI + Uvicorn |
| **Validation** | Pydantic v2 |
| **Testing** | pytest (293 tests), pytest-mock |
| **ML metrics** | scikit-learn (accuracy, precision, recall, F1) |
| **HTTP client** | requests (with retry/backoff) |
| **API test client** | httpx (FastAPI TestClient) |

---

## API Reference

### `GET /health`

Returns service status and model availability.

**Response:** `200 OK`

```json
{ "status": "ok", "service": "uk-employment-law-change-detector", "model_available": true }
```

### `POST /analyse`

Runs the full legislation monitoring pipeline.

**Request body:**

```json
{
  "url": "https://www.legislation.gov.uk/ukpga/1996/18",
  "min_confidence": 0.0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | legislation.gov.uk URL for a monitored act |
| `min_confidence` | float | No (default: 0.0) | Minimum classifier confidence to include a change |

**Responses:**

| Status | Condition |
|--------|-----------|
| `200` | Pipeline completed successfully |
| `422` | Invalid URL or unrecognised act |
| `502` | Failed to fetch XML from legislation.gov.uk |
| `503` | No trained model available |

---

## About This Project

UK employment legislation changes frequently. For large organisations with in-house legal departments, staying compliant is manageable. For the 5.5 million SMEs that make up 99.9% of UK businesses, it is not. A single missed change to unfair dismissal rules, minimum wage thresholds, or working time regulations can result in tribunal claims, financial penalties, and reputational damage.

This project is a standalone prototype that tackles that problem at the data and ML layer — automatically detecting, classifying, and structuring legislative changes into actionable alerts that non-specialists can understand. It implements the end-to-end flow from raw legislation XML to structured compliance alerts.

### What this prototype demonstrates

- **Real data integration** — direct connection to the UK Government's legislation.gov.uk API, parsing Crown Legislation Markup Language (CLML) XML
- **ML-driven classification** — fine-tuned BERT model distinguishing material compliance changes from minor editorial amendments with confidence scoring
- **Structured, actionable output** — each detected change is enriched with domain classification, effective dates, penalty thresholds, affected business types, and obligation categorisation
- **Production-ready API design** — FastAPI endpoint with request validation, error handling, and a baseline-then-diff workflow suitable for scheduled monitoring
- **Comprehensive test coverage** — 293 automated tests across all pipeline stages
