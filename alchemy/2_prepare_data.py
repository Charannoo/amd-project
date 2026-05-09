"""
STEP 2: Prepare training data from BindingDB
Run: python 2_prepare_data.py
Takes ~10-15 minutes

Reads the BindingDB TSV (uncompressed .tsv or compressed .tsv.gz) and produces
train.jsonl + val.jsonl with protein sequences, pKi values, and binary labels.
"""
import csv
import gzip
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

DATA_DIR = Path.home() / "alchemy_training" / "data"

# ── Locate the BindingDB file ────────────────────────────────────────────────
candidates = [
    DATA_DIR / "bindingdb_raw.tsv",        # uncompressed (from .zip extraction)
    DATA_DIR / "bindingdb_raw.tsv.gz",      # legacy .gz format
]
# Also check for any BindingDB TSV that might have been extracted
for p in DATA_DIR.glob("BindingDB_All*.tsv"):
    candidates.insert(0, p)

INPUT_FILE = None
for c in candidates:
    if c.is_file() and c.stat().st_size > 1_000_000:
        INPUT_FILE = c
        break

if INPUT_FILE is None:
    print("❌ ERROR: BindingDB data file not found in ~/alchemy_training/data/")
    print("   Expected one of:")
    for c in candidates:
        print(f"     {c}")
    print("\n   Run 'bash 1_setup.sh' first, or download manually from:")
    print("   https://www.bindingdb.org/rwd/bind/chemsearch/marvin/SDFdownload.jsp?all_download=yes")
    sys.exit(1)

print(f"=== Loading BindingDB from {INPUT_FILE.name} ({INPUT_FILE.stat().st_size / 1e9:.2f} GB) ===")


# ── Open file (handles both .gz and plain .tsv) ─────────────────────────────
def open_tsv(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


# ── Parse and filter ────────────────────────────────────────────────────────
rows = []
skipped = Counter()

with open_tsv(INPUT_FILE) as f:
    reader = csv.DictReader(f, delimiter="\t")

    # Verify expected columns exist
    if reader.fieldnames is None:
        print("❌ ERROR: Could not read TSV headers. File may be corrupted.")
        sys.exit(1)

    has_ki = "Ki (nM)" in reader.fieldnames
    seq_col = next((c for c in reader.fieldnames if c.startswith("BindingDB Target Chain Sequence")), None)
    has_seq = seq_col is not None
    has_smi = "Ligand SMILES" in reader.fieldnames

    if not has_seq:
        print(f"❌ ERROR: No 'BindingDB Target Chain Sequence' column found")
        print(f"   Available columns ({len(reader.fieldnames)}):")
        for col in reader.fieldnames[:20]:
            print(f"     - {col}")
        sys.exit(1)

    if not has_ki:
        has_ic50 = "IC50 (nM)" in reader.fieldnames
        if has_ic50:
            print("⚠ 'Ki (nM)' column not found; using 'IC50 (nM)' as fallback")
        else:
            print("❌ ERROR: Neither 'Ki (nM)' nor 'IC50 (nM)' column found")
            sys.exit(1)

    ki_column = "Ki (nM)" if has_ki else "IC50 (nM)"
    print(f"   Using affinity column: '{ki_column}'")
    print(f"   Sequence column: '{seq_col}'")
    print(f"   Ligand column: 'Ligand SMILES'")
    print()

    for i, row in enumerate(reader):
        if i % 200_000 == 0 and i > 0:
            print(f"  Scanned {i:,} rows, kept {len(rows):,} ...")

        # ── Extract fields ──
        sequence = row.get(seq_col, "").strip()
        smiles = row.get("Ligand SMILES", "").strip()
        ki_raw = row.get(ki_column, "").strip()

        # ── Quality filters ──
        if not sequence:
            skipped["no_sequence"] += 1
            continue
        if not smiles:
            skipped["no_smiles"] += 1
            continue
        if not ki_raw:
            skipped["no_affinity"] += 1
            continue

        # Sequence length: ESM2 max is 1022 tokens; skip very short proteins
        if len(sequence) < 50:
            skipped["sequence_too_short"] += 1
            continue
        if len(sequence) > 1022:
            skipped["sequence_too_long"] += 1
            continue

        # Skip very large ligands (unlikely drug-like)
        if len(smiles) > 200:
            skipped["smiles_too_long"] += 1
            continue

        # ── Parse Ki/IC50 → pKi ──
        try:
            # Remove inequality prefixes like ">", "<", ">=", "~"
            cleaned = ki_raw.replace(">", "").replace("<", "").replace("=", "").replace("~", "").strip()
            ki_nm = float(cleaned)
            if ki_nm <= 0:
                skipped["negative_ki"] += 1
                continue
            ki_m = ki_nm * 1e-9  # nM → M
            pki = -math.log10(ki_m)  # pKi = -log10(Ki_M)

            # Keep pharmacologically relevant range: pKi 4–12 (10 mM to 1 pM)
            if not (4.0 <= pki <= 12.0):
                skipped["pki_out_of_range"] += 1
                continue
        except (ValueError, OverflowError):
            skipped["parse_error"] += 1
            continue

        # ── Keep row ──
        rows.append({
            "sequence": sequence,
            "smiles": smiles,
            "pki": round(pki, 3),
            # Label: 1 = active binder (pKi >= 6 → Ki <= 1 µM), 0 = weak/inactive
            "label": 1 if pki >= 6.0 else 0,
        })

        # Soft cap at 300K rows to keep training time reasonable (~6-8 hrs)
        if len(rows) >= 300_000:
            print(f"  Reached 300K row cap at row {i:,}")
            break

# ── Report statistics ────────────────────────────────────────────────────────
active = sum(r["label"] for r in rows)
inactive = len(rows) - active

print(f"\n{'='*60}")
print(f"  Total clean rows: {len(rows):,}")
print(f"  Active binders (pKi≥6, Ki≤1µM): {active:,} ({100*active/max(1,len(rows)):.1f}%)")
print(f"  Weak binders:                    {inactive:,} ({100*inactive/max(1,len(rows)):.1f}%)")
print(f"{'='*60}")

if skipped:
    print(f"\n  Skip reasons:")
    for reason, count in sorted(skipped.items(), key=lambda x: -x[1]):
        print(f"    {reason}: {count:,}")

if len(rows) < 10_000:
    print(f"\n⚠ WARNING: Only {len(rows):,} rows kept. This may not be enough for good training.")
    print("  Consider relaxing filters or checking the BindingDB file format.")

if len(rows) == 0:
    print("\n❌ ERROR: No valid rows found. Cannot proceed.")
    sys.exit(1)

# ── Shuffle and split 90/10 ──────────────────────────────────────────────────
random.seed(42)
random.shuffle(rows)

split = int(len(rows) * 0.9)
train_rows = rows[:split]
val_rows = rows[split:]

# ── Write JSONL ──────────────────────────────────────────────────────────────
train_file = DATA_DIR / "train.jsonl"
val_file = DATA_DIR / "val.jsonl"

with open(train_file, "w", encoding="utf-8") as f:
    for r in train_rows:
        f.write(json.dumps(r) + "\n")

with open(val_file, "w", encoding="utf-8") as f:
    for r in val_rows:
        f.write(json.dumps(r) + "\n")

# ── Verify files ─────────────────────────────────────────────────────────────
train_size = train_file.stat().st_size / 1e6
val_size = val_file.stat().st_size / 1e6

print(f"\n  ✅ {len(train_rows):,} train rows → {train_file} ({train_size:.1f} MB)")
print(f"  ✅ {len(val_rows):,}  val rows   → {val_file} ({val_size:.1f} MB)")

# Quick sanity check: read first row back
with open(train_file) as f:
    sample = json.loads(f.readline())
    print(f"\n  Sample row:")
    print(f"    sequence: {sample['sequence'][:60]}... ({len(sample['sequence'])} aa)")
    print(f"    smiles:   {sample['smiles'][:60]}")
    print(f"    pki:      {sample['pki']}")
    print(f"    label:    {sample['label']}")

print(f"\n{'='*60}")
print(f"  Data ready. Run: python 3_train.py")
print(f"{'='*60}")
