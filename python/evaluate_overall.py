import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from core.recognizer import get_recognizer


MYA_TO_LATIN = str.maketrans("\u1040\u1041\u1042\u1043\u1044\u1045\u1046\u1047\u1048\u1049", "0123456789")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _to_text(value):
    if value is None:
        return ""
    return str(value).strip()


def normalize_nrc(value):
    text = _to_text(value)
    if not text:
        return ""
    text = text.translate(MYA_TO_LATIN)
    text = text.lower()
    text = text.replace("\u1014\u102d\u102f\u1004\u103a", "n")
    text = text.replace("(n)", "n")
    text = text.replace("\u1014\u102d\u102f\u1004\u1039", "n")
    text = re.sub(r"[\s\(\)\[\]\{\}/\\\-_.:]+", "", text)
    return text


def normalize_date(value):
    text = _to_text(value)
    if not text:
        return ""
    text = text.translate(MYA_TO_LATIN)
    parts = re.findall(r"\d+", text)
    if len(parts) >= 3:
        day = str(int(parts[0])) if parts[0] else "0"
        mon = str(int(parts[1])) if parts[1] else "0"
        year = parts[2]
        if len(year) == 2:
            year = f"20{year}"
        year = str(int(year)) if year else "0"
        return f"{int(day):02d}-{int(mon):02d}-{int(year):04d}"
    if len(parts) == 1 and len(parts[0]) == 8:
        d = parts[0][0:2]
        m = parts[0][2:4]
        y = parts[0][4:8]
        return f"{d}-{m}-{y}"
    return "".join(parts)


def normalize_blood(value):
    text = _to_text(value)
    if not text:
        return ""
    low = text.lower().replace(" ", "")

    # Burmese variants
    if "\u1021\u1031\u1018\u102e" in text or "\u1031\u1021\u1018\u102e" in text:
        return "AB"
    if "\u1021\u1031" in text or "\u1031\u1021" in text:
        return "A"
    if "\u1018\u102e" in text:
        return "B"
    if "\u1021\u102d\u102f" in text or "\u1021\u102f\u102d" in text:
        return "O"

    # Latin variants
    up = re.sub(r"[^a-z0-9+]", "", low.upper())
    if "AB" in up:
        return "AB"
    if up in {"A", "A+", "A-"} or up.startswith("A"):
        return "A"
    if up in {"B", "B+", "B-"} or up.startswith("B"):
        return "B"
    if up in {"O", "0", "O+", "O-"} or up.startswith("O"):
        return "O"
    return up


def _row_get(row, *keys):
    for key in keys:
        if key in row and _to_text(row[key]):
            return _to_text(row[key])
    return ""


def _resolve_image_path(image_value, manifest_path):
    raw = _to_text(image_value)
    if not raw:
        return ""

    p = Path(raw)
    if p.is_absolute():
        return str(p.resolve())

    candidates = [
        (manifest_path.parent / p).resolve(),
        (PROJECT_ROOT / p).resolve(),
        (Path.cwd() / p).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Fallback to project-root-relative form for clearer debugging output.
    return str((PROJECT_ROOT / p).resolve())


def load_manifest_csv(csv_path):
    csv_path = Path(csv_path)
    rows = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            image = _row_get(r, "image_path", "image", "path", "img")
            if not image:
                continue
            rows.append(
                {
                    "image_path": _resolve_image_path(image, csv_path),
                    "nrc": _row_get(r, "nrc", "nrcNumber", "nrc_number"),
                    "birth_date": _row_get(r, "birth_date", "birthDate", "dob"),
                    "issue_date": _row_get(r, "issue_date", "issueDate"),
                    "blood": _row_get(r, "blood", "bloodType", "blood_type"),
                }
            )
    return rows


def evaluate(manifest_rows, conf_threshold=0.25):
    recognizer = get_recognizer()

    field_total = defaultdict(int)
    field_ok = defaultdict(int)
    strict_total = 0
    strict_ok = 0

    detail = []

    for idx, row in enumerate(manifest_rows, start=1):
        image_path = row["image_path"]
        if not Path(image_path).exists():
            print(f"[skip] missing image: {image_path}")
            continue

        try:
            image = cv2.imread(image_path)
        except FileNotFoundError:
            print(f"[skip] missing image: {image_path}")
            continue

        if image is None:
            print(f"[skip] unreadable image: {image_path}")
            continue

        pred = recognizer.recognize(image, conf_threshold=conf_threshold)

        pred_nrc = _to_text(pred.get("nrcNumberBurmese") or pred.get("nrcNumber"))
        pred_birth = _to_text(pred.get("birthDate") or pred.get("birthDateLatin"))
        pred_issue = _to_text(pred.get("issueDate") or pred.get("issueDateLatin"))
        pred_blood = _to_text(pred.get("bloodType"))

        gt_nrc = _to_text(row.get("nrc"))
        gt_birth = _to_text(row.get("birth_date"))
        gt_issue = _to_text(row.get("issue_date"))
        gt_blood = _to_text(row.get("blood"))

        checks = {}

        if gt_nrc:
            field_total["nrc"] += 1
            checks["nrc"] = normalize_nrc(pred_nrc) == normalize_nrc(gt_nrc)
            field_ok["nrc"] += int(checks["nrc"])
        else:
            checks["nrc"] = None

        if gt_birth:
            field_total["birth_date"] += 1
            checks["birth_date"] = normalize_date(pred_birth) == normalize_date(gt_birth)
            field_ok["birth_date"] += int(checks["birth_date"])
        else:
            checks["birth_date"] = None

        if gt_issue:
            field_total["issue_date"] += 1
            checks["issue_date"] = normalize_date(pred_issue) == normalize_date(gt_issue)
            field_ok["issue_date"] += int(checks["issue_date"])
        else:
            checks["issue_date"] = None

        if gt_blood:
            field_total["blood"] += 1
            checks["blood"] = normalize_blood(pred_blood) == normalize_blood(gt_blood)
            field_ok["blood"] += int(checks["blood"])
        else:
            checks["blood"] = None

        active_checks = [v for v in checks.values() if v is not None]
        if active_checks:
            strict_total += 1
            strict_pass = all(active_checks)
            strict_ok += int(strict_pass)
        else:
            strict_pass = None

        detail.append(
            {
                "image_path": image_path,
                "gt_nrc": gt_nrc,
                "pred_nrc": pred_nrc,
                "nrc_ok": checks["nrc"],
                "gt_birth_date": gt_birth,
                "pred_birth_date": pred_birth,
                "birth_date_ok": checks["birth_date"],
                "gt_issue_date": gt_issue,
                "pred_issue_date": pred_issue,
                "issue_date_ok": checks["issue_date"],
                "gt_blood": gt_blood,
                "pred_blood": pred_blood,
                "blood_ok": checks["blood"],
                "strict_ok": strict_pass,
                "nrc_conf": float(pred.get("confidence", 0.0) or 0.0),
                "birth_conf": float(pred.get("birthDateConfidence", 0.0) or 0.0),
                "issue_conf": float(pred.get("issueDateConfidence", 0.0) or 0.0),
                "blood_conf": float(pred.get("bloodTypeConfidence", 0.0) or 0.0),
            }
        )

        if idx % 25 == 0:
            print(f"[progress] {idx}/{len(manifest_rows)} evaluated")

    summary = {
        "samples_evaluated": len(detail),
        "strict_total": strict_total,
        "strict_correct": strict_ok,
        "strict_accuracy": (strict_ok / strict_total) if strict_total else 0.0,
        "field_accuracy": {
            field: (field_ok[field] / field_total[field]) if field_total[field] else 0.0
            for field in ("nrc", "birth_date", "issue_date", "blood")
        },
        "field_total": dict(field_total),
        "field_correct": dict(field_ok),
    }

    return summary, detail


def save_detail_csv(detail, out_csv):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not detail:
        out_csv.write_text("", encoding="utf-8")
        return
    fieldnames = list(detail[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail)


def save_summary_json(summary, out_json):
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_accuracy_bar(summary, out_png):
    labels = ["strict", "nrc", "birth_date", "issue_date", "blood"]
    values = [
        summary.get("strict_accuracy", 0.0),
        summary.get("field_accuracy", {}).get("nrc", 0.0),
        summary.get("field_accuracy", {}).get("birth_date", 0.0),
        summary.get("field_accuracy", {}).get("issue_date", 0.0),
        summary.get("field_accuracy", {}).get("blood", 0.0),
    ]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, [v * 100 for v in values])
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Overall System Accuracy")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{value*100:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_blood_confusion(detail, out_png):
    labels = ["A", "B", "AB", "O", "OTHER"]
    index = {k: i for i, k in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)

    def to_bucket(v):
        n = normalize_blood(v)
        return n if n in {"A", "B", "AB", "O"} else "OTHER"

    used = 0
    for row in detail:
        gt = _to_text(row.get("gt_blood"))
        if not gt:
            continue
        pred = _to_text(row.get("pred_blood"))
        gi = index[to_bucket(gt)]
        pi = index[to_bucket(pred)]
        mat[gi, pi] += 1
        used += 1

    if used == 0:
        return

    plt.figure(figsize=(6, 5))
    plt.imshow(mat, interpolation="nearest")
    plt.title("Blood Type Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")

    thresh = mat.max() / 2 if mat.max() > 0 else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(
                j,
                i,
                str(mat[i, j]),
                ha="center",
                va="center",
                color="white" if mat[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate overall OCR system accuracy (NRC + birth_date + issue_date + blood) "
            "from a CSV manifest."
        )
    )
    parser.add_argument("--manifest", required=True, help="CSV with columns: image_path,nrc,birth_date,issue_date,blood")
    parser.add_argument("--out_dir", default="results/overall_eval", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_manifest_csv(args.manifest)
    print(f"Loaded manifest rows: {len(rows)}")
    summary, detail = evaluate(rows, conf_threshold=args.conf)

    detail_csv = out_dir / "detail.csv"
    summary_json = out_dir / "summary.json"
    acc_png = out_dir / "accuracy_bar.png"
    blood_png = out_dir / "blood_confusion.png"

    save_detail_csv(detail, detail_csv)
    save_summary_json(summary, summary_json)
    plot_accuracy_bar(summary, acc_png)
    plot_blood_confusion(detail, blood_png)

    print("\n=== OVERALL RESULTS ===")
    print(f"Strict Accuracy: {summary['strict_accuracy']*100:.2f}% ({summary['strict_correct']}/{summary['strict_total']})")
    for field in ("nrc", "birth_date", "issue_date", "blood"):
        acc = summary["field_accuracy"].get(field, 0.0) * 100
        ok = summary["field_correct"].get(field, 0)
        total = summary["field_total"].get(field, 0)
        print(f"{field}: {acc:.2f}% ({ok}/{total})")
    print("\nSaved:")
    print(f"- {detail_csv}")
    print(f"- {summary_json}")
    print(f"- {acc_png}")
    print(f"- {blood_png}")


if __name__ == "__main__":
    main()
