#!/usr/bin/env python3
"""
Fetch oncology drug reimbursement data from the SÚKL API.
Outputs structured JSON for the static frontend.

API docs: https://prehledy.sukl.cz/docs/?url=/dlp.api.json#/
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

API_BASE = "https://prehledy.sukl.cz/dlp/v1"
ATC_PREFIXES = ("L01", "L02", "V10X", "V10B")
MAX_WORKERS = 10
REQUEST_DELAY = 0.05  # seconds between requests per thread

# --- Cancer type keyword mapping for filtering ---

CANCER_KEYWORDS = {
    "NSCLC": ["NSCLC", "nemalobuněčn"],
    "SCLC": [r"(?<!N)SCLC", r"(?<!ne)malobuněčn"],
    "Karcinom prsu": [
        "prsu", "mammae",
        "fulvestrant", r"inhibitor\w* aromatáz", "letrozolem", "anastrozolem",
        "tamoxifen",
    ],
    "Melanom": ["melanom"],
    "Karcinom ledvin": [
        "karcinom ledvin", "karcinomem ledvin", "karcinomu ledvin",
        "renáln", r"\bRCC\b",
    ],
    "Kolorektální karcinom": [
        "kolorektáln", "tlusté střevo", "tlustého střeva",
        "FOLFIRI", "FOLFOX", r"\bmCRC\b",
    ],
    "Karcinom vaječníků": ["vaječník", "ovari"],
    "Karcinom prostaty": ["prostat", "kastračně rezistentní", r"\bmCRPC\b"],
    "Karcinom žaludku": ["žaludk", "gastro-ezofageáln", "gastroezofageáln", "GEJ"],
    "Karcinom jícnu": ["jícn", "ezofag"],
    "Karcinom hlavy a krku": ["hlavy a krku", "HNSCC"],
    "Uroteliální karcinom": ["měchýř", "urotel"],
    "Hepatocelulární karcinom": ["hepatocelulárn", r"\bHCC\b"],
    "Karcinom děložního hrdla": ["děložního hrdla", "cervix", "cervik"],
    "Karcinom endometria": ["endometri"],
    "Karcinom štítné žlázy": ["karcinom štítné žlázy", "karcinomem štítné žlázy"],
    "Karcinom pankreatu": [r"(?:karcinom\w*|adenokarcinom\w*|nádor\w*)\s+pankrea"],
    "Cholangiokarcinom": ["cholangiokarci", "žlučových cest", "žlučovodů"],
    "GIST": [r"(?<!vyjma )\bGIST\b", "gastrointestinální stromální"],
    "Hodgkinův lymfom": [r"(?<!ne)(?<!non-)hodgkin", "autologní transplantac"],
    "Non-Hodgkinův lymfom": [
        "non-Hodgkin", "DLBCL", r"folikulárn\w+ lymfom", r"plášťov\w+ buněk",
        r"velkobuněčn\w+.{0,30}lymfom", "MALT", "marginální zón",
        "Burkitt",
    ],
    "CLL/SLL": [r"chronick\w+ lymf\w+ leuk", r"\bCLL\b"],
    "Mnohočetný myelom": ["myelom"],
    "AML": ["akutní myeloidní", r"\bAML\b"],
    "CML": ["chronická myeloidní", r"\bCML\b", "Philadelphia"],
    "Mezoteliom": ["mezoteliom"],
    "Sarkom": ["sarkom"],
    "Neuroblastom": ["neuroblastom"],
    "Gliom": ["glioblastom", "gliom"],
    "MDS": ["myelodysplastick"],
    "Waldenstromova makroglobulinemie": ["Waldenström", "makroglobulinemi"],
    "Mastocytóza": ["mastocytóz"],
    "Merkelův karcinom": ["Merkel"],
    "Thymom": ["thymom"],
    "Neuroendokrinní nádory": ["neuroendokrinn", "karcinoid"],
    "ITP": ["trombocytopenická purpura", r"\bITP\b"],
}

IMMUNOTHERAPY_ATC = ("L01FF", "L01FX04", "L01FX18")


def fetch_json(url, retries=3):
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                return None


def fetch_drug_detail(kod):
    time.sleep(REQUEST_DELAY)
    return fetch_json(f"{API_BASE}/cau-scau/{kod}")


def is_oncology(detail):
    if not detail:
        return False
    atc = detail.get("ATCkod", "")
    return any(atc.startswith(p) for p in ATC_PREFIXES)


def classify_type(atc_code):
    if any(atc_code.startswith(p) for p in IMMUNOTHERAPY_ATC):
        return "Imunoterapie"
    if atc_code.startswith("V10"):
        return "Radioligandová terapie"
    if atc_code.startswith("L02"):
        return "Hormonální terapie"
    return "Cílená léčba"


import re as _re

# Patterns that start the "common conditions" section — not actual indications
def extract_cancer_types(text):
    if not text:
        return []
    found = []
    text_lower = text.lower()
    for cancer, keywords in CANCER_KEYWORDS.items():
        for kw in keywords:
            # Use regex if the keyword contains regex metacharacters
            if "\\" in kw or "(?<" in kw:
                if _re.search(kw, text, _re.IGNORECASE):
                    found.append(cancer)
                    break
            else:
                if kw.lower() in text_lower:
                    found.append(cancer)
                    break
    return found if found else ["Ostatní"]


def _normalize(text):
    return " ".join(text.split()).strip()


# Regex to find the start of common/shared conditions
_COMMON_BLOCK_RE = _re.compile(
    r"(?:\n|(?<=\.))\s*(?:"
    r"Pro úhradu ve všech|Pro všechny indikace|Pro obě indikace|"
    r"Pro úhradu (?:ve výše|v uveden|ve všech výše)|"
    r"Podmínkou léčby (?:ve všech|v obou)|"
    r"Pro úhradu musí být|Podmínkou úhrady ve všech|"
    r"Pacienti? (?:ve všech|v obou|kumulativně)|"
    r"Jedná se o pacienty? ve velmi|"
    r"Léčba \w+ je hrazena do|"
    r"Pro indikace uvedené v bodech|"
    r"Pro všechny výše uvedené)",
    _re.IGNORECASE,
)


_ROMAN_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6,
              "VII": 7, "VIII": 8, "IX": 9, "X": 10}


def _find_sequential_matches(matches, get_value, expected_first=1):
    """Filter matches that form a sequential run starting at expected_first."""
    sequential = []
    expected = expected_first
    for m in matches:
        v = get_value(m)
        if v == expected:
            sequential.append(m)
            expected += 1
    return sequential


def _find_top_sections(text):
    """Detect top-level Roman (I., II., ...) or letter (A., B., ...) sections.

    Sections may start at line beginning, after a colon, or after a sentence
    boundary (period+space). Sequence filter ensures false matches inside
    prose ("methoda I. typu") are dropped unless they form a real run.
    Returns list of section body strings, or None if not found.
    """
    # Roman numerals (require trailing space + lowercase verb-like start to
    # cut down on false matches mid-sentence)
    roman_pat = _re.compile(
        r'(?:^|[\n.:;])\s*(X|IX|VI{0,3}|IV|I{1,3})\.\s+(?=[a-zěščřžýáíéůú])',
        _re.MULTILINE,
    )
    matches = list(roman_pat.finditer(text))
    seq = _find_sequential_matches(matches, lambda m: _ROMAN_MAP.get(m.group(1)))
    if len(seq) >= 2:
        return _slice_sections(text, seq)

    # Letter sections: "A. ", "B. ", "C. " — same constraints
    letter_pat = _re.compile(
        r'(?:^|[\n.:;])\s*([A-Z])\.\s+(?=[a-zěščřžýáíéůú])',
        _re.MULTILINE,
    )
    matches = list(letter_pat.finditer(text))
    seq = _find_sequential_matches(
        matches, lambda m: ord(m.group(1)) - ord('A') + 1
    )
    if len(seq) >= 2:
        return _slice_sections(text, seq)

    return None


def _slice_sections(text, matches):
    """Given matches at section boundaries, return (header, body) per section.

    Header is the marker + first phrase up to first sub-item or sentence end.
    Body is everything else in that section.
    """
    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip().rstrip(";").rstrip(".")
        sections.append(section_text)
    return sections


def _find_subitems(text):
    """Find numbered sub-items 1) 2) ... or 1. 2. ... in text.

    Matches digit+marker tokens that are not preceded by another digit or
    word char. Sequential filter ensures false matches mid-prose are dropped
    unless they form a real run.
    """
    # "1) 2) ..." pattern — must be a standalone marker, not part of a word
    pat = _re.compile(r'(?<![\d\w])(\d+)\)\s+(?=[a-zěščřžýáíéůú])', _re.MULTILINE)
    matches = list(pat.finditer(text))
    seq = _find_sequential_matches(matches, lambda m: int(m.group(1)))
    if len(seq) >= 2:
        return seq

    # "1. 2. ..." pattern
    dot_pat = _re.compile(
        r'(?:^|[:\n]|(?<=\.\s))\s*(\d+)\.\s+(?=[a-zěščřžýáíéůú])',
        _re.MULTILINE,
    )
    matches = list(dot_pat.finditer(text))
    seq = _find_sequential_matches(matches, lambda m: int(m.group(1)))
    if len(seq) >= 2:
        return seq

    return []


def _split_section_by_subitems(section_text):
    """Split a section into items by numbered sub-items, prepending parent prefix.

    Returns list of item texts. If no sub-items found, returns [section_text].
    """
    matches = _find_subitems(section_text)
    if not matches:
        return [section_text.strip()]

    # Pre-text is everything before the first sub-item (parent context)
    prefix = section_text[:matches[0].start()].strip().rstrip(":").rstrip()

    # Strip trailing common conditions block (applies to all sub-items)
    text_for_items = section_text
    obecne_in_tail = None
    last_start = matches[-1].start()
    after_last = section_text[last_start:]
    cm = _COMMON_BLOCK_RE.search(after_last)
    if cm:
        obecne_in_tail = after_last[cm.start():].strip()
        text_for_items = section_text[:last_start + cm.start()]
        matches = [m for m in matches if m.start() < len(text_for_items)]

    items = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text_for_items)
        item_body = text_for_items[start:end].strip().rstrip(";").rstrip()
        # Prepend parent context so each sub-item carries its cancer-type heading
        if prefix:
            full = f"{prefix} {item_body}"
        else:
            full = item_body
        items.append(full)

    return items, obecne_in_tail


def _split_indication_text(text, uhrada, platnost_do):
    """Split a multi-indication text into individual indication items.

    Strategy:
      1) Detect top-level Roman/letter sections at line start. If present,
         each section becomes its own indication and is recursively split
         by numbered sub-items (1), 2)) with the section heading preserved.
      2) Otherwise, split by numbered sub-items at any level, prepending
         the parent prefix (text before the first item) to each item.
      3) Otherwise, return the text as a single item.
    """
    # 1) Top-level Roman/letter sections
    sections = _find_top_sections(text)
    if sections:
        items = []
        common = None
        for sec in sections:
            res = _split_section_by_subitems(sec)
            if isinstance(res, tuple):
                sub_items, sec_common = res
                if sec_common and not common:
                    common = sec_common
            else:
                sub_items = res
            for it in sub_items:
                items.append({
                    "text": it,
                    "uhrada": uhrada,
                    "platnostDo": platnost_do,
                    "obecnePodminky": None,
                })
        if common:
            for it in items:
                it["obecnePodminky"] = common
        return items

    # 2) Numbered sub-items at top level — preserve prefix as parent context
    res = _split_section_by_subitems(text)
    if isinstance(res, tuple):
        sub_items, common = res
    else:
        sub_items, common = res, None

    if len(sub_items) >= 2:
        return [{
            "text": it,
            "uhrada": uhrada,
            "platnostDo": platnost_do,
            "obecnePodminky": common,
        } for it in sub_items]

    # 3) Single item
    return [{
        "text": text.strip(),
        "uhrada": uhrada,
        "platnostDo": platnost_do,
    }]


def process_drugs(details):
    grouped = {}

    for d in details:
        if not d or not is_oncology(d):
            continue

        name = d["nazev"]
        atc = d.get("ATCkod", "")

        if name not in grouped:
            grouped[name] = {
                "nazev": name,
                "atcKod": atc,
                "typ": classify_type(atc),
                "formy": set(),
                "baleni": [],
                "_raw_uhrady": [],
                "_kodSUKL": d.get("kodSUKL", ""),
            }

        drug = grouped[name]
        sila = d.get("sila", "")
        forma = d.get("lekovaFormaKod", "")
        if sila and forma:
            drug["formy"].add(f"{sila} {forma}")

        # Per-package info — pick the highest úhrada across this package's
        # úhrada entries (typically just one) as the package's reimbursement.
        pkg_uhrada = 0
        for u in d.get("uhrady", []):
            val = u.get("uhrada") or 0
            if val > pkg_uhrada:
                pkg_uhrada = val
        drug["baleni"].append({
            "kodSUKL": d.get("kodSUKL", ""),
            "sila": sila,
            "lekovaForma": forma,
            "baleni": d.get("baleni", ""),
            "doplnek": d.get("doplnek", ""),
            "uhrada": pkg_uhrada,
            "cenaPuvodce": d.get("cenaPuvodce"),
            "maxCenaLekarna": d.get("maxCenaLekarna"),
        })

        forma_str = f"{d.get('sila', '')} {d.get('lekovaFormaKod', '')}".strip()
        for u in d.get("uhrady", []):
            text = u.get("indikacniOmezeni")
            if not text:
                continue
            drug["_raw_uhrady"].append({
                "text": text,
                "uhrada": u.get("uhrada") or 0,
                "platnostDo": u.get("platnostDocasneUhrady"),
                "_forma": forma_str,
            })

    def _process_raw_to_indikace(raw_entries):
        """Split, dedup, and build indication list from raw úhrada entries."""
        all_items = []
        for entry in raw_entries:
            items = _split_indication_text(
                entry["text"], entry["uhrada"], entry["platnostDo"]
            )
            all_items.extend(items)

        # Deduplicate. Two items collapse only when their normalized text is
        # nearly identical — distinct sub-items (e.g. "ve druhé linii" vs.
        # "ve třetí linii") are preserved. Across multiple SUKL packages of
        # the same drug, identical indication texts collapse to one entry
        # (keeping the highest úhrada).
        def _dedup_key(text):
            # Strip all non-alphanumeric chars for dedup so cross-package
            # variations (whitespace, "+" vs " ", punctuation) collapse.
            return _re.sub(r"[^\w]+", "", _normalize(text).lower())

        seen = {}
        for item in all_items:
            key = _dedup_key(item["text"])
            if key in seen:
                if item["uhrada"] > seen[key]["uhrada"]:
                    seen[key] = item
            else:
                seen[key] = item

        indikace = []
        for item in seen.values():
            cancer_types = extract_cancer_types(item["text"])
            indikace.append({
                "text": item["text"],
                "uhrada": item["uhrada"],
                "platnostDo": item["platnostDo"],
                "obecnePodminky": item.get("obecnePodminky"),
                "typyNadoru": cancer_types,
            })
        indikace.sort(key=lambda x: x["typyNadoru"][0] if x["typyNadoru"] else "")
        return indikace

    result = []
    for drug in grouped.values():
        raw = drug.pop("_raw_uhrady")

        # Separate formulations: IV (INF CNC SOL) vs SC (INJ SOL)
        iv_raw = []
        sc_raw = []
        for entry in raw:
            if "INJ SOL" in entry.get("_forma", ""):
                sc_raw.append(entry)
            else:
                iv_raw.append(entry)

        # Process main (IV or all if no SC)
        main_raw = iv_raw if sc_raw else raw
        indikace = _process_raw_to_indikace(main_raw)

        # Remove redundant unsplit blocks
        if len(indikace) > 1:
            split_types = set()
            for ind in indikace:
                if len(ind["typyNadoru"]) <= 2:
                    split_types.update(ind["typyNadoru"])
            indikace = [
                ind for ind in indikace
                if not (len(ind["typyNadoru"]) > 3
                        and len(ind["text"]) > 500
                        and set(ind["typyNadoru"]) <= split_types | {"Ostatní"})
            ]

        drug["indikace"] = indikace

        if not indikace:
            continue

        drug["formy"] = sorted(drug["formy"])
        # Sort packages: by strength (numeric if possible), then by package text
        drug["baleni"] = sorted(
            drug.get("baleni", []),
            key=lambda b: (b.get("lekovaForma", ""), b.get("sila", ""), b.get("baleni", "")),
        )
        all_types = set()
        for ind in drug["indikace"]:
            all_types.update(ind["typyNadoru"])
        drug["typyNadoru"] = sorted(all_types)
        # Split packages into IV/SC matching the indication split, if SC variant exists
        if sc_raw:
            iv_baleni = [b for b in drug["baleni"] if "INJ SOL" not in b.get("lekovaForma", "")]
            sc_baleni = [b for b in drug["baleni"] if "INJ SOL" in b.get("lekovaForma", "")]
            drug["baleni"] = iv_baleni
        result.append(drug)

        # Create separate SC drug entry if applicable
        if sc_raw:
            sc_indikace = _process_raw_to_indikace(sc_raw)
            if sc_indikace:
                sc_drug = {
                    "nazev": f"{drug['nazev']} (SC)",
                    "atcKod": drug["atcKod"],
                    "typ": drug["typ"],
                    "formy": sorted(f for f in drug["formy"] if "INJ SOL" in f),
                    "baleni": sc_baleni,
                    "_kodSUKL": drug["_kodSUKL"],
                    "indikace": sc_indikace,
                }
                all_types_sc = set()
                for ind in sc_indikace:
                    all_types_sc.update(ind["typyNadoru"])
                sc_drug["typyNadoru"] = sorted(all_types_sc)
                result.append(sc_drug)

    return sorted(result, key=lambda x: x["nazev"])


def _fetch_substance(kod_sukl):
    """Fetch the active substance name for a SUKL code."""
    time.sleep(REQUEST_DELAY)
    slozeni = fetch_json(f"{API_BASE}/slozeni/{kod_sukl}")
    if not slozeni:
        return None
    # Find the first active substance (kodSlozeni == "L" = léčivá látka)
    for item in slozeni:
        if item.get("kodSlozeni") == "L":
            latka = fetch_json(
                f"{API_BASE}/ciselnik-latky?kodLatky={item['kodLatky']}"
            )
            if latka:
                name = latka.get("nazev", "")
                return name.capitalize() if name else None
    return None


def fetch_substance_names(drugs):
    """Fetch active substance names for all drugs."""
    print(f"Stahování názvů účinných látek ({len(drugs)} léků)...")
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {}
        for drug in drugs:
            kod = drug.get("_kodSUKL", "")
            if kod:
                futures[ex.submit(_fetch_substance, kod)] = drug
        for f in as_completed(futures):
            drug = futures[f]
            name = f.result()
            if name:
                drug["ucinnaLatka"] = name
            done += 1
            if done % 50 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")

    # Clean up internal field
    for drug in drugs:
        drug.pop("_kodSUKL", None)


def main():
    print("=== SÚKL: Úhrady onkologických léčiv ===\n")

    # 1. Batch metadata
    print("Načítání metadat...")
    batches = fetch_json(f"{API_BASE}/aktualni-davky")
    if not batches:
        print("Chyba: nelze načíst metadata z API.")
        sys.exit(1)
    scau = next((b for b in batches if b["typ"] == "SCAU"), None)
    if scau:
        print(f"  SCAU platnost od: {scau['platnostOd']}, verze: {scau['verze']}")

    # 2. All reimbursed drug codes
    print("Načítání seznamu hrazených léčiv...")
    codes = fetch_json(f"{API_BASE}/lecive-pripravky?uvedeneCeny=false&typSeznamu=scau")
    if not codes:
        print("Chyba: nelze načíst seznam léčiv.")
        sys.exit(1)
    print(f"  Celkem {len(codes)} hrazených léčiv")

    # 3. Fetch details concurrently
    print(f"Stahování detailů ({MAX_WORKERS} vláken)...")
    details = []
    done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_drug_detail, k): k for k in codes}
        for f in as_completed(futures):
            r = f.result()
            if r:
                details.append(r)
            done += 1
            if done % 500 == 0 or done == len(codes):
                print(f"  {done}/{len(codes)}")

    print(f"  Úspěšně staženo: {len(details)}")

    # 4. Filter & process
    onco_count = sum(1 for d in details if is_oncology(d))
    print(f"  Onkologických (ATC L01*): {onco_count}")

    drugs = process_drugs(details)
    print(f"  Seskupeno do {len(drugs)} léků")

    # 5. Fetch substance names
    fetch_substance_names(drugs)

    # 5b. Merge in radioligands (not in SCAU; curated reference data)
    radio_path = Path(__file__).parent / "data" / "radioligands.json"
    if radio_path.exists():
        with open(radio_path, "r", encoding="utf-8") as f:
            radio = json.load(f)
        radio_drugs = radio.get("drugs", [])
        drugs.extend(radio_drugs)
        drugs.sort(key=lambda x: x["nazev"])
        print(f"  + {len(radio_drugs)} radioligandů (ze static reference)")

    # Collect filter values
    all_cancers = set()
    all_types = set()
    for drug in drugs:
        all_cancers.update(drug.get("typyNadoru", []))
        all_types.add(drug["typ"])

    # 5. Save
    output = {
        "metadata": {
            "datumAktualizace": datetime.now().strftime("%Y-%m-%d"),
            "scauPlatnostOd": scau["platnostOd"][:10] if scau else None,
            "scauVerze": scau["verze"] if scau else None,
            "pocetLeku": len(drugs),
            "pocetZaznamu": onco_count,
        },
        "filtry": {
            "typyTerapie": sorted(all_types),
            "typyNadoru": sorted(all_cancers),
        },
        "leky": drugs,
    }

    out = Path(__file__).parent / "data" / "drugs.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total_ind = sum(len(d["indikace"]) for d in drugs)
    print(f"\nHotovo! Uloženo do {out}")
    print(f"  {len(drugs)} léků, {total_ind} indikací")


if __name__ == "__main__":
    main()
