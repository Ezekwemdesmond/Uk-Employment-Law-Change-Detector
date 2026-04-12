Project: UK Employment Law Change Detector
Purpose: Portfolio project for UK Innovator Founder Visa — demonstrates core regulatory monitoring pipeline for complyAI, an AI-powered compliance SaaS for UK SMEs.
Stack: Python 3.12.7, Windows/VS Code, HuggingFace Transformers, spaCy, BeautifulSoup4, FastAPI for serving, pytest for testing.
Architecture:
- Layer 1: legislation.gov.uk XML scraper + diff engine
- Layer 2: Fine-tuned BERT binary classifier (material vs minor change)
- Layer 3: Structured JSON output (domain, effective date, penalty threshold, affected business types, obligation type)
Key conventions:
- All data stored under /data (raw XML, diffs, labelled examples)
- All models stored under /models
- All outputs stored under /outputs
- Write docstrings for every function
- Write a pytest test for every module
- Keep a running CHANGELOG.md updated with every significant addition
Target Acts to monitor: Employment Rights Act 1996, National Minimum Wage Act 1998, Equality Act 2010, Working Time Regulations 1998.