"""
Training data builder for the material-change classifier.

Generates a labelled dataset with synthetic examples covering both classes:
  - 0 = minor change (formatting, cross-references, punctuation, renumbering)
  - 1 = material change (new obligations, penalties, scope changes, definitions)

Outputs a CSV to data/labelled/ with columns: text, label, change_type, act_name.
"""

import csv
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATA_LABELLED_DIR = Path(__file__).parent.parent / "data" / "labelled"

# The four target acts
ACT_NAMES = [
    "employment_rights_act_1996",
    "national_minimum_wage_act_1998",
    "equality_act_2010",
    "working_time_regulations_1998",
]

# ---------------------------------------------------------------------------
# Templates for synthetic training examples
# ---------------------------------------------------------------------------

# Material change templates (label = 1)
MATERIAL_TEMPLATES: dict[str, list[str]] = {
    "new_obligation": [
        "The employer must now provide written particulars of employment within {days} days of the start date.",
        "Employers are required to maintain records of working hours for all employees for a minimum of {years} years.",
        "A new duty is imposed on employers to consult with employee representatives before making redundancies.",
        "The employer shall ensure that all workers receive a written statement of their rights under this Act.",
        "Employers must provide itemised pay statements to all workers, including those on zero-hours contracts.",
        "A new obligation requires employers to report gender pay gap data annually to the Secretary of State.",
        "The employer is now required to provide reasonable adjustments for disabled employees within {days} days of a request.",
        "Employers must conduct risk assessments for pregnant workers and new mothers within {weeks} weeks.",
        "A duty is placed on employers to prevent harassment by third parties in the workplace.",
        "The employer shall provide whistleblowing protection to all workers who report wrongdoing in good faith.",
    ],
    "penalty_change": [
        "The maximum penalty for non-compliance has been increased from {old_amount} to {new_amount}.",
        "Employers who fail to comply may be fined up to {amount} per employee affected.",
        "The tribunal may now award compensation of up to {weeks} weeks' pay for unfair dismissal.",
        "A new civil penalty of up to {amount} applies to employers who fail to pay the national minimum wage.",
        "The penalty for unlawful deduction from wages has been increased to {amount}.",
        "Employers found guilty of discrimination may face unlimited compensation awards.",
        "The maximum award for whistleblowing detriment has been raised to {amount}.",
        "A daily penalty of {daily_amount} applies for each day of continued non-compliance.",
        "The basic award for unfair dismissal is now calculated at {weeks} weeks' pay.",
        "Aggravated damages of up to {amount} may be awarded in cases of deliberate discrimination.",
    ],
    "scope_extension": [
        "The provisions of this section now apply to workers as well as employees.",
        "The definition of 'worker' is extended to include gig economy and platform workers.",
        "These protections are now extended to agency workers from day one of their assignment.",
        "The Act now covers employees working remotely or from home.",
        "Protection against unfair dismissal is extended to employees with {months} months' service.",
        "The right to request flexible working is extended to all employees from the first day of employment.",
        "Parental leave provisions now apply to foster parents and kinship carers.",
        "The scope of this regulation is extended to include small businesses with fewer than {count} employees.",
        "Apprentices are now included within the scope of minimum wage provisions.",
        "The working time regulations are extended to cover previously excluded transport workers.",
    ],
    "new_definition": [
        "A new definition of 'employee' includes those with worker status in the gig economy.",
        "'Sexual harassment' is now defined to include conduct of a sexual nature that creates a hostile environment.",
        "The term 'reasonable adjustments' is defined to include modifications to working arrangements.",
        "'Continuous employment' is now defined to include breaks of up to {weeks} weeks.",
        "A new definition of 'disability' aligns with the social model of disability.",
        "'Working time' is defined to include time spent on-call at the employer's premises.",
        "The definition of 'redundancy' is expanded to include restructuring situations.",
        "'Victimisation' is defined to include any detriment arising from a protected act.",
        "A new definition of 'harassment' covers unwanted conduct related to any protected characteristic.",
        "'Night worker' is defined as a worker who works at least {hours} hours during the night period.",
    ],
    "new_exemption": [
        "An exemption is introduced for employers with fewer than {count} employees during the first {years} years.",
        "Small businesses are exempt from the reporting requirement if turnover is below {amount}.",
        "The {days}-day notice period does not apply to employees on probationary contracts.",
        "Religious organisations are exempt from certain provisions relating to belief discrimination.",
        "A new exemption applies to genuine occupational requirements in specific circumstances.",
        "Employers may be exempt from certain duties where compliance would cause disproportionate burden.",
        "The minimum wage provisions do not apply to voluntary workers as newly defined.",
        "An exemption is granted for family businesses where all workers are close relatives.",
        "Start-up companies are exempt from gender pay gap reporting for {years} years.",
        "Emergency service workers are exempt from maximum working time provisions.",
    ],
}

# Minor change templates (label = 0)
MINOR_TEMPLATES: dict[str, list[str]] = {
    "cross_reference": [
        "In section {old_num}, for 'section {ref_old}' substitute 'section {ref_new}'.",
        "The reference to Schedule {old_num} is updated to Schedule {new_num}.",
        "For 'paragraph {old_para}' read 'paragraph {new_para}' throughout.",
        "Cross-references to the 1995 Act are updated to refer to the 2010 Act.",
        "In subsection ({sub}), the reference to regulation {old_reg} is corrected to regulation {new_reg}.",
        "References to 'the 1998 Regulations' are substituted with 'the Working Time Regulations 1998'.",
        "The citation 'SI 1999/1234' is corrected to 'SI 1999/4321'.",
        "In paragraph {para}, for 'Part I' substitute 'Part 1'.",
        "The reference to Article {old_art} of the Treaty is updated to Article {new_art}.",
        "Schedule {sch} paragraph {para} is renumbered as Schedule {sch} paragraph {new_para}.",
    ],
    "punctuation_fix": [
        "In section {num}, after 'employer' insert a comma.",
        "The semicolon at the end of paragraph ({para}) is replaced with a full stop.",
        "In subsection ({sub}), omit the unnecessary comma after 'employee'.",
        "The hyphen in 'full-time' is removed to read 'full time'.",
        "Quotation marks are corrected to single quotes throughout section {num}.",
        "In the definition of 'worker', a missing apostrophe is inserted.",
        "The colon after 'following' is replaced with a dash.",
        "Parentheses are added around the phrase 'as amended'.",
        "The comma splice in paragraph {para} is corrected with a semicolon.",
        "In subsection ({sub}), 'ie' is corrected to 'i.e.'.",
    ],
    "hyperlink_update": [
        "The URL for guidance is updated from http://example.gov.uk to https://www.gov.uk/guidance.",
        "Online resources are now hosted at the new government domain.",
        "The link to the explanatory memorandum is updated to the current location.",
        "Web addresses are updated to use HTTPS throughout.",
        "The reference to the online calculator is updated with the new URL.",
        "Links to archived legislation are updated to the National Archives.",
        "The HMRC guidance URL is corrected to the current web address.",
        "External links are updated following website migration.",
        "The hyperlink to the ACAS code of practice is refreshed.",
        "Digital service URLs are updated to reflect the new GOV.UK structure.",
    ],
    "renumbering": [
        "Section {old_num} is renumbered as section {new_num}.",
        "Paragraphs ({old_para}) to ({end_para}) are renumbered as ({new_para}) to ({new_end}).",
        "Part {old_part} becomes Part {new_part} following consolidation.",
        "The Schedule is renumbered from Schedule {old_sch} to Schedule {new_sch}.",
        "Subsections are renumbered consecutively following the repeal of subsection ({sub}).",
        "Regulation {old_reg} is renumbered as regulation {new_reg}.",
        "Article numbers are updated to reflect the insertion of new Article {art}.",
        "Chapter {old_ch} is redesignated as Chapter {new_ch}.",
        "The table of contents is updated to reflect new section numbers.",
        "Paragraph numbering is corrected following editorial review.",
    ],
    "formatting_change": [
        "Italic text is converted to standard formatting throughout.",
        "Bold headings are standardised across all sections.",
        "The table in Schedule {sch} is reformatted for accessibility.",
        "Numbered lists are converted to lettered lists for consistency.",
        "Margin notes are incorporated into the main text.",
        "The font size in tables is standardised to match body text.",
        "White space is adjusted between paragraphs.",
        "Indentation is corrected in subsection ({sub}).",
        "Column widths in the table are adjusted for readability.",
        "Section headings are reformatted in title case.",
    ],
}


def _fill_template(template: str) -> str:
    """
    Fill placeholder values in a template string with random realistic values.

    Args:
        template: String containing placeholders like {days}, {amount}, etc.

    Returns:
        Template with all placeholders replaced.
    """
    replacements = {
        "days": str(random.choice([7, 14, 28, 30, 60, 90])),
        "weeks": str(random.choice([2, 4, 8, 12, 26, 52])),
        "months": str(random.choice([1, 3, 6, 12, 24])),
        "years": str(random.choice([1, 2, 3, 5, 6])),
        "hours": str(random.choice([3, 4, 5, 6])),
        "count": str(random.choice([5, 10, 20, 50, 250])),
        "amount": f"£{random.choice([1000, 5000, 10000, 20000, 50000, 100000]):,}",
        "old_amount": f"£{random.choice([500, 1000, 2500, 5000]):,}",
        "new_amount": f"£{random.choice([10000, 20000, 50000, 100000]):,}",
        "daily_amount": f"£{random.choice([100, 200, 500, 1000]):,}",
        "num": str(random.randint(1, 200)),
        "old_num": str(random.randint(1, 100)),
        "new_num": str(random.randint(101, 200)),
        "ref_old": str(random.randint(1, 50)),
        "ref_new": str(random.randint(51, 100)),
        "old_para": random.choice(["a", "b", "c", "d", "i", "ii", "iii"]),
        "new_para": random.choice(["e", "f", "g", "h", "iv", "v", "vi"]),
        "end_para": random.choice(["d", "e", "f", "g"]),
        "new_end": random.choice(["h", "i", "j", "k"]),
        "para": random.choice(["a", "b", "c", "1", "2", "3"]),
        "sub": str(random.randint(1, 10)),
        "old_reg": str(random.randint(1, 20)),
        "new_reg": str(random.randint(21, 40)),
        "old_art": str(random.randint(1, 50)),
        "new_art": str(random.randint(51, 100)),
        "sch": str(random.randint(1, 10)),
        "old_sch": str(random.randint(1, 5)),
        "new_sch": str(random.randint(6, 10)),
        "old_part": random.choice(["I", "II", "III", "1", "2", "3"]),
        "new_part": random.choice(["IV", "V", "VI", "4", "5", "6"]),
        "old_ch": str(random.randint(1, 5)),
        "new_ch": str(random.randint(6, 10)),
        "art": str(random.randint(1, 20)),
    }

    result = template
    for key, value in replacements.items():
        result = result.replace("{" + key + "}", value)
    return result


def generate_examples(
    n_per_class: int = 100,
    seed: int | None = 42,
) -> list[dict[str, Any]]:
    """
    Generate synthetic labelled examples for both classes.

    Args:
        n_per_class: Number of examples to generate per class (material/minor).
        seed: Random seed for reproducibility. Pass None for non-deterministic.

    Returns:
        List of dicts with keys: text, label, change_type, act_name.
    """
    if seed is not None:
        random.seed(seed)

    examples: list[dict[str, Any]] = []

    # Generate material change examples (label = 1)
    material_types = list(MATERIAL_TEMPLATES.keys())
    for i in range(n_per_class):
        change_type = material_types[i % len(material_types)]
        templates = MATERIAL_TEMPLATES[change_type]
        template = random.choice(templates)
        text = _fill_template(template)
        act_name = random.choice(ACT_NAMES)
        examples.append({
            "text": text,
            "label": 1,
            "change_type": change_type,
            "act_name": act_name,
        })

    # Generate minor change examples (label = 0)
    minor_types = list(MINOR_TEMPLATES.keys())
    for i in range(n_per_class):
        change_type = minor_types[i % len(minor_types)]
        templates = MINOR_TEMPLATES[change_type]
        template = random.choice(templates)
        text = _fill_template(template)
        act_name = random.choice(ACT_NAMES)
        examples.append({
            "text": text,
            "label": 0,
            "change_type": change_type,
            "act_name": act_name,
        })

    # Shuffle to mix classes
    random.shuffle(examples)

    logger.info(
        "Generated %d examples: %d material, %d minor",
        len(examples), n_per_class, n_per_class,
    )
    return examples


def examples_from_diff(
    diff: dict[str, Any],
    default_label: int | None = None,
) -> list[dict[str, Any]]:
    """
    Convert a preprocessed diff object into unlabelled (or default-labelled) examples.

    This is useful for preparing real diffs for inference or manual labelling.

    Args:
        diff: Preprocessed diff dict from preprocessor.preprocess_diff().
        default_label: Optional label to assign. None leaves label unset.

    Returns:
        List of example dicts ready for classification or labelling.
    """
    act_name = diff.get("act_name", "unknown")
    examples: list[dict[str, Any]] = []

    # Added sections
    for section_id, text in diff.get("added", {}).items():
        ex = {
            "text": text,
            "change_type": "added_section",
            "act_name": act_name,
            "section_id": section_id,
        }
        if default_label is not None:
            ex["label"] = default_label
        examples.append(ex)

    # Removed sections
    for section_id, text in diff.get("removed", {}).items():
        ex = {
            "text": text,
            "change_type": "removed_section",
            "act_name": act_name,
            "section_id": section_id,
        }
        if default_label is not None:
            ex["label"] = default_label
        examples.append(ex)

    # Modified sections — use the combined text
    for section_id, data in diff.get("modified", {}).items():
        combined = data.get("combined", f"{data.get('old', '')} {data.get('new', '')}")
        ex = {
            "text": combined,
            "change_type": "modified_section",
            "act_name": act_name,
            "section_id": section_id,
        }
        if default_label is not None:
            ex["label"] = default_label
        examples.append(ex)

    return examples


def save_dataset(
    examples: list[dict[str, Any]],
    filename: str | None = None,
    output_dir: Path | None = None,
) -> Path:
    """
    Save a list of examples to a CSV file.

    Args:
        examples: List of example dicts (must have at least 'text' key).
        filename: Output filename. Defaults to timestamped name.
        output_dir: Directory to save to. Defaults to data/labelled.

    Returns:
        Path to the saved CSV file.
    """
    target_dir = output_dir if output_dir is not None else DATA_LABELLED_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"labelled_dataset_{timestamp}.csv"

    filepath = target_dir / filename

    # Determine fieldnames from the first example, ensuring consistent order
    fieldnames = ["text", "label", "change_type", "act_name"]
    # Add any extra fields found in examples
    for ex in examples:
        for key in ex:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(examples)

    logger.info("Saved %d examples to %s", len(examples), filepath)
    return filepath


def load_dataset(filepath: Path | str) -> list[dict[str, Any]]:
    """
    Load a labelled dataset from CSV.

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of example dicts.
    """
    filepath = Path(filepath)
    examples: list[dict[str, Any]] = []

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert label to int if present
            if "label" in row and row["label"]:
                row["label"] = int(row["label"])
            examples.append(dict(row))

    logger.info("Loaded %d examples from %s", len(examples), filepath)
    return examples


def build_training_dataset(
    n_per_class: int = 100,
    seed: int = 42,
    output_dir: Path | None = None,
) -> Path:
    """
    Generate and save a complete training dataset.

    Convenience function that generates synthetic examples and saves them.

    Args:
        n_per_class: Examples per class (total = 2 * n_per_class).
        seed: Random seed for reproducibility.
        output_dir: Override output directory.

    Returns:
        Path to the saved CSV file.
    """
    examples = generate_examples(n_per_class=n_per_class, seed=seed)
    return save_dataset(examples, output_dir=output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = build_training_dataset(n_per_class=100)
    print(f"Dataset saved to: {path}")
