# Changelog

All notable changes to the UK Employment Law Change Detector are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [0.4.0] — 2026-04-13

### Added — Render deployment & HuggingFace model loading

#### Deployment
- `render.yaml` — Render web service config: Python 3.12.7, `pip install -r requirements.txt`
  build command, `uvicorn src.api:app --host 0.0.0.0 --port $PORT` start command.
- `Procfile` — Heroku-compatible process declaration for `uvicorn` (also read by Render).
- API deployed and publicly accessible at
  `https://uk-employment-law-change-detector.onrender.com`.

#### `src/classifier.py`
- `load_model()` now loads `bert-base-uncased` directly from HuggingFace Hub instead of
  from a local `models/best_model/` checkpoint. Weights are downloaded on first call and
  cached in `models/hub/`; subsequent calls are served entirely from the local cache.
- Added module-level constants `HF_MODEL_NAME` and `HF_CACHE_DIR`.

#### `src/api.py`
- `MODELS_DIR` updated to point at `models/` (root) rather than `models/best_model/`.
- `_model_is_available()` now checks for a populated `models/hub/` HuggingFace cache
  directory instead of a `config.json` fine-tuned checkpoint.
- Updated `_run_scorer` error message to reflect automatic HuggingFace download behaviour.

#### Documentation
- `README.md` — added "Live API" section with the public Render URL, health-check,
  interactive-docs, and analyse-endpoint links.

---

## [0.3.0] — 2026-04-12

### Added — Layer 3: Structured JSON Output Pipeline & API

#### `src/domain_classifier.py`
- `classify_domain(act_name)` — maps canonical act names to compliance domains
  (`employment`, `wage`, `equality`, `working_time`).
- `classify_domain_from_url(url)` — infers domain from legislation.gov.uk URL path.
- `classify_domain_from_text(text)` — keyword-based domain inference from free text.
- `url_to_act_name(url)` — reverse-maps a URL to a canonical act name.

#### `src/penalty_extractor.py`
- `extract_penalty(text)` — regex-based extraction of monetary penalty thresholds;
  recognises `£N,NNN`, per-day/per-week/per-employee qualifiers, and `unlimited`
  penalty phrases; returns `None` when no threshold is found.
- `extract_all_amounts(text)` — returns all monetary amounts found in text.

#### `src/output_builder.py`
- `extract_effective_date(text)` — extracts effective dates from contextual phrases
  (`with effect from`, `commencing`, ISO dates, named-month dates).
- `extract_affected_business_types(text)` — identifies affected business categories
  via keyword matching; defaults to `["all employers"]`.
- `infer_obligation_type(change_type, text)` — maps change type + text signals to
  one of five obligation types: `new_duty`, `changed_duty`, `removed_duty`,
  `changed_penalty`, `extended_scope`.
- `build_material_record(prediction, act_name)` — constructs a single structured
  output record with all five required fields.
- `build_output(scored_result, min_confidence)` — filters to material predictions,
  builds records for all, returns the full structured output dict.
- `save_output(output, act_name, output_dir)` — persists output to timestamped JSON.

#### `src/api.py` — FastAPI application
- `GET /health` — service status and model-availability check.
- `POST /analyse` — full pipeline endpoint:
  1. Validates legislation.gov.uk URL.
  2. Identifies act from URL.
  3. Fetches current XML from legislation.gov.uk.
  4. Compares against most recent cached baseline in `data/raw/`.
  5. First call saves baseline and returns `pipeline_status="baseline_established"`.
  6. Subsequent calls: diff → preprocess → BERT classify → build structured output.
  7. Returns `pipeline_status` of `"changes_found"` or `"no_changes"`.
- Returns `503` if no trained model is available, `502` on network failures.

#### Tests
- `tests/test_domain_classifier.py` — 22 tests covering all four mapping functions.
- `tests/test_penalty_extractor.py` — 24 tests covering unlimited penalties,
  monetary amounts, qualifiers, edge cases, and `extract_all_amounts`.
- `tests/test_output_builder.py` — 35 tests covering date extraction, business-type
  detection, obligation inference, record building, and JSON output.
- `tests/test_api.py` — 20 tests covering health check, request validation, baseline
  flow, network errors, model-unavailable 503, and full pipeline with mocked scorer.

#### Project scaffolding
- `requirements.txt` — added `fastapi`, `uvicorn`, `httpx`.

---

## [0.2.0] — 2026-04-12

### Added — Layer 2: ML Classification Pipeline

#### `src/data_builder.py`
- `generate_examples(n_per_class, seed)` — generates synthetic labelled training
  examples for material (class 1) and minor (class 0) changes using template-based
  generation with realistic UK employment law language.
- `examples_from_diff(diff, default_label)` — converts preprocessed diff objects
  into example dicts ready for classification or labelling.
- `save_dataset(examples, filename, output_dir)` — saves labelled examples to CSV
  with columns: `text`, `label`, `change_type`, `act_name`.
- `load_dataset(filepath)` — loads a labelled CSV back into memory.
- `build_training_dataset(n_per_class, seed, output_dir)` — convenience function
  to generate and save a complete training dataset.
- **Material change templates:** new obligations, penalty changes, scope extensions,
  new definitions, new exemptions (50 templates total).
- **Minor change templates:** cross-references, punctuation fixes, hyperlink updates,
  renumbering, formatting changes (50 templates total).

#### `src/classifier.py`
- `LegislationDataset` — PyTorch Dataset wrapper for tokenized examples.
- `split_dataset(examples, ratios, seed)` — stratified 80/10/10 train/val/test split.
- `compute_metrics(labels, predictions)` — calculates accuracy, precision, recall, F1.
- `evaluate(model, dataloader, device)` — evaluates model on a dataset and returns
  metrics and loss.
- `train(dataset_path, config, output_dir)` — full fine-tuning loop:
  - Loads `bert-base-uncased` from HuggingFace
  - Uses AdamW optimizer with linear warmup scheduler
  - Tracks validation metrics per epoch
  - Saves best checkpoint to `models/best_model/`
  - Logs training history to `outputs/training_log.json`
- `load_model(model_path)` — loads a trained model and tokenizer.
- **Default config:** batch_size=16, lr=2e-5, epochs=3, max_length=256.

#### `src/scorer.py`
- `Scorer` class — wraps a trained model for inference:
  - `predict_single(text)` — returns probability scores and confidence flag.
  - `predict_batch(texts)` — batch inference with configurable batch size.
  - `predict_examples(examples)` — merges predictions into example dicts.
  - `predict_diff(diff)` — scores all sections in a preprocessed diff object.
- `save_predictions(predictions, filename, output_dir)` — saves results as JSON.
- `score_diff(diff, model_path, save)` — convenience function for scoring diffs.
- `score_texts(texts, model_path, save)` — convenience function for raw text scoring.
- **Confidence threshold:** 0.75 — predictions below this are flagged for human review.

#### Tests
- `tests/test_data_builder.py` — 35 tests covering template filling, example
  generation, diff conversion, CSV save/load roundtrips.
- `tests/test_classifier.py` — 25 tests covering dataset wrapper, split logic,
  metrics computation, and config validation.
- `tests/test_scorer.py` — 30 tests covering Scorer initialization, single/batch
  prediction, confidence flagging, and JSON output.

#### Project scaffolding
- `requirements.txt` — added `torch`, `transformers`, `scikit-learn`.
- `data/labelled/` — output directory for generated training data.

---

## [0.1.0] — 2026-04-12

### Added — Layer 1: Data Ingestion Pipeline

#### `src/scraper.py`
- `build_session()` — constructs a `requests.Session` with exponential-backoff
  retry logic (3 attempts, status codes 429/500/502/503/504).
- `fetch_legislation_xml(session, act_path)` — fetches the CLML XML feed for a
  given act path from `legislation.gov.uk/…/data.xml`; handles 429 rate-limit
  responses by honouring the `Retry-After` header.
- `save_xml(act_name, xml_content, output_dir)` — persists raw XML to
  `data/raw/<act_name>_<YYYYMMDDTHHMMSSz>.xml`.
- `scrape_all(acts, output_dir)` — orchestrates scraping of all four target acts
  with a 1.5 s inter-request delay; failed acts are logged and omitted from the
  return value.
- **Target acts configured:** Employment Rights Act 1996, National Minimum Wage
  Act 1998, Equality Act 2010, Working Time Regulations 1998.

#### `src/diff_engine.py`
- `parse_xml(xml_source)` — parses XML from a string, bytes, or `Path`.
- `extract_sections(root)` — walks a CLML element tree and returns a
  `section_id -> text` dict for all section-level elements (`Section`,
  `P1group`, `Part`, `Chapter`, `Schedule`, `Article`, `Regulation`, `Rule`).
- `compute_diff(old_xml, new_xml, act_name)` — compares two document versions
  and returns a structured diff object with `added`, `removed`, and `modified`
  sections plus a `summary` count block.
- `save_diff(diff, act_name, output_dir)` — serialises a diff to
  `data/diffs/<act_name>_<YYYYMMDDTHHMMSSz>.json`.

#### `src/preprocessor.py`
- `strip_xml_tags(text)` — removes all XML/HTML markup via BeautifulSoup/lxml.
- `remove_citation_markers(text)` — strips CLML editorial markers (`[F1]`,
  `[E+W]`, etc.) and commencement notes.
- `expand_abbreviations(text)` — expands UK legal abbreviations (Reg, ERA, WTR,
  s. 12, i.e., etc.) for improved tokenisation.
- `normalise_whitespace(text)` — collapses all whitespace to single spaces.
- `truncate_for_bert(text, max_chars)` — truncates at sentence boundaries to fit
  within BERT's 512-token limit.
- `preprocess_text(raw_text, truncate)` — full pipeline combining the above.
- `preprocess_diff(diff, truncate)` — applies `preprocess_text` to every text
  field in a diff object; adds a `combined` field (`old [SEP] new`) for
  modified sections.

#### Tests
- `tests/test_scraper.py` — 14 tests covering session config, XML fetching,
  rate-limit handling, file saving, and `scrape_all` orchestration.
- `tests/test_diff_engine.py` — 18 tests covering XML parsing, section
  extraction (including namespaced CLML), diff computation, and JSON output.
- `tests/test_preprocessor.py` — 25 tests covering each pipeline stage and the
  full `preprocess_text` / `preprocess_diff` integration.

#### Project scaffolding
- `requirements.txt` — pinned dependencies for Layer 1 (`requests`, `lxml`,
  `beautifulsoup4`, `pytest`).
- `pytest.ini` — test runner configuration.
- `data/raw/`, `data/diffs/` — created with `.gitkeep` placeholders.
