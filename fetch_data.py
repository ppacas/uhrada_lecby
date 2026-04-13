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
ATC_PREFIXES = ("L01", "L02")
MAX_WORKERS = 10
REQUEST_DELAY = 0.05  # seconds between requests per thread

# --- Cancer type keyword mapping for filtering ---

CANCER_KEYWORDS = {
    "NSCLC": ["NSCLC", "nemalobuněčn"],
    "SCLC": [r"(?<!N)SCLC", r"(?<!ne)malobuněčn"],
    "Karcinom prsu": [
        "prsu", "mammae",
        "fulvestrant", "inhibitor\w* aromatáz", "letrozolem", "anastrozolem",
        "tamoxifen",
    ],
    "Melanom": ["melanom"],
    "Karcinom ledvin": [
        "karcinom ledvin", "karcinomem ledvin", "karcinomu ledvin",
        "renáln", r"\bRCC\b",
    ],
    "Kolorektální karcinom": ["kolorektáln", "tlusté střevo", "tlustého střeva"],
    "Karcinom vaječníků": ["vaječník", "ovari"],
    "Karcinom prostaty": ["prostat"],
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


def _split_indication_text(text, uhrada, platnost_do):
    """Split a multi-indication text into individual indication items.

    Returns list of dicts with: text, uhrada, platnostDo, obecnePodminky.
    Handles numbering patterns: 1), 2)... or 1., 2.... or A), B)...
    """
    # Try multiple numbering patterns
    # Try "1), 2)" pattern first
    pat = _re.compile(r'(?:^|[\n:;])\s*\d+\)\s*', _re.MULTILINE)
    matches = list(pat.finditer(text))
    if len(matches) < 2:
        matches = []

    # Try "N. " pattern (e.g. "1. k léčbě...  2. v léčbě...") if nothing found
    if len(matches) < 2:
        dot_pat = _re.compile(r'(?:^|[:\n]|(?<=\.\s))\s*(\d+)\.\s+', _re.MULTILINE)
        candidates = list(dot_pat.finditer(text))
        if len(candidates) >= 2:
            sequential = []
            expected = 1
            for m in candidates:
                num = int(m.group(1))
                if num == expected:
                    sequential.append(m)
                    expected += 1
                elif num > expected:
                    break
            if len(sequential) >= 2:
                matches = sequential

    # Try "A) ... B) ... C)" pattern — must be sequential A, B, C...
    if len(matches) < 2:
        letter_pat = _re.compile(r'(?:^|[\n:.])\s*([A-Z])\)\s*', _re.MULTILINE)
        candidates = list(letter_pat.finditer(text))
        if len(candidates) >= 2:
            sequential = []
            expected_ord = ord('A')
            for m in candidates:
                if ord(m.group(1)) == expected_ord:
                    sequential.append(m)
                    expected_ord += 1
            if len(sequential) >= 2:
                matches = sequential

    # Try Roman numerals: "I. ... II. ... III."
    if len(matches) < 2:
        roman_vals = [("I", 1), ("II", 2), ("III", 3), ("IV", 4), ("V", 5), ("VI", 6)]
        roman_pat = _re.compile(
            r'(?:^|[\n:.])\s*(I{1,3}V?|IV|VI{0,3})\.\s+',
            _re.MULTILINE,
        )
        candidates = list(roman_pat.finditer(text))
        roman_map = {r: v for r, v in roman_vals}
        if len(candidates) >= 2:
            sequential = []
            expected = 1
            for m in candidates:
                val = roman_map.get(m.group(1))
                if val == expected:
                    sequential.append(m)
                    expected += 1
            if len(sequential) >= 2:
                matches = sequential

    if len(matches) < 2:
        return [{
            "text": text.strip(),
            "uhrada": uhrada,
            "platnostDo": platnost_do,
        }]

    # Extract common conditions block first
    obecne = ""
    text_for_items = text
    last_item_start = matches[-1].start()
    after_last = text[last_item_start:]
    cm = _COMMON_BLOCK_RE.search(after_last)
    if cm:
        obecne = after_last[cm.start():].strip()
        text_for_items = text[:last_item_start + cm.start()]
        # Re-find matches in trimmed text
        matches = [m for m in matches if m.start() < len(text_for_items)]

    # Extract each numbered item
    items = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text_for_items)
        item_text = text_for_items[start:end].strip().rstrip(";")

        items.append({
            "text": item_text,
            "uhrada": uhrada,
            "platnostDo": platnost_do,
            "obecnePodminky": obecne if obecne else None,
        })

    return items


def _item_fingerprint(text):
    """Extract key clinical concepts for dedup matching.

    Extracts the treatment modality and key nouns, ignoring filler words.
    Two items describing the same treatment for the same cancer should
    produce the same fingerprint.
    """
    norm = _normalize(text).lower()
    # Extract key clinical terms
    terms = set()
    # Treatment modality
    for kw in ["monoterapii", "kombinaci", "kombinované", "adjuvantní",
               "neoadjuvantní", "udržovací", "první lini", "1. lini"]:
        if kw in norm:
            terms.add(kw.replace("ované", "").replace("aci", ""))
    # Cancer type / drug combo keywords — synonyms map to canonical term
    cancer_kws = {
        "melanom": "melanom", "renáln": "rcc", "ledvin": "rcc",
        "nsclc": "nsclc", "nemalobuněčn": "nsclc",
        "urotel": "urotel", "kolorektáln": "crc",
        "jícn": "jicen", "žaludk": "zaludek", "gastro": "zaludek",
        "prsu": "prsu", "hlavy a krku": "hnscc", "hodgkin": "hodgkin",
        "myelom": "myelom", "vaječník": "ovarium", "prostat": "prostata",
        "ipilimumab": "ipilimumab", "kabozantinib": "kabozantinib",
        "lenvatinib": "lenvatinib", "cisplatin": "cisplatin",
        "bevacizumab": "bevacizumab", "trastuzumab": "trastuzumab",
        "pemetrexed": "pemetrexed", "paklitaxel": "paklitaxel",
        "gemcitabin": "gemcitabin", "autologní transplant": "autotx",
        "etoposid": "etoposid",
    }
    for kw, canonical in cancer_kws.items():
        if kw in norm:
            terms.add(canonical)
    return frozenset(terms)


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
                "_raw_uhrady": [],
                "_kodSUKL": d.get("kodSUKL", ""),
            }

        drug = grouped[name]
        sila = d.get("sila", "")
        forma = d.get("lekovaFormaKod", "")
        if sila and forma:
            drug["formy"].add(f"{sila} {forma}")

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

        # Deduplicate by fingerprint, keep highest price
        seen = {}
        for item in all_items:
            fp = _item_fingerprint(item["text"])
            if fp in seen:
                if item["uhrada"] > seen[fp]["uhrada"]:
                    seen[fp] = item
            else:
                seen[fp] = item

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
        all_types = set()
        for ind in drug["indikace"]:
            all_types.update(ind["typyNadoru"])
        drug["typyNadoru"] = sorted(all_types)
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
