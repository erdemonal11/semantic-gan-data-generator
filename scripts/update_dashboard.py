import json
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

PROCESSED_DIR = Path("data/processed")
SYNTHETIC_DIR = Path("data/synthetic")
WEB_SOURCE_DIR = Path("src")

REAL_TRIPLES_PATH = PROCESSED_DIR / "kg_triples_ids.txt"
MAPPINGS_PATH = PROCESSED_DIR / "kg_mappings.json"
LOSS_PLOT_PATH = PROCESSED_DIR / "loss_curve.png"

OUTPUT_JSON = WEB_SOURCE_DIR / "dashboard_data.json"
OUTPUT_PLOT_WEB = WEB_SOURCE_DIR / "loss_curve.png"

def main():
    print("[INFO] Updating Website Dashboard Data...")

    if LOSS_PLOT_PATH.exists():
        shutil.copy(LOSS_PLOT_PATH, OUTPUT_PLOT_WEB)
        print("[INFO] Loss curve updated.")

    print("[INFO] Loading ID Mappings...")
    with open(MAPPINGS_PATH, "r", encoding="utf-8") as f:
        id_to_name = json.load(f)

    if not list(SYNTHETIC_DIR.glob("generated_*.txt")):
        print("[WARN] No generated data found.")
        return

    latest_file = sorted(SYNTHETIC_DIR.glob("generated_*.txt"))[-1]
    print(f"[INFO] Processing latest file: {latest_file.name}")

    synthetic_triples = []
    with open(latest_file, "r") as f:
        next(f) 
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                r, t, score = parts[0], parts[1], float(parts[2])
                synthetic_triples.append((r, t, score))

    real_pairs = set()
    with open(REAL_TRIPLES_PATH, "r") as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) == 3:
                real_pairs.add(f"{p[1]}|{p[2]}")

    novel_count = 0
    decoded_hypotheses = []


    for i, (r, t, score) in enumerate(synthetic_triples):
        

        key = f"{r}|{t}"
        is_novel = key not in real_pairs
        if is_novel:
            novel_count += 1


        if i < 50:
            tail_name = id_to_name.get(t, t) 
            

            rel_name = r.replace("dblp:", "").replace("inYear", "Published in").replace("publishedIn", "Venue:").replace("wrote", "Authored Paper")
            
            decoded_hypotheses.append({
                "relation": rel_name,
                "entity": tail_name,
                "score": f"{score:.4f}",
                "novel": is_novel
            })


    total = len(synthetic_triples)
    novelty_score = (novel_count / total) * 100 if total > 0 else 0
    uniqueness_score = (len(set([x[1] for x in synthetic_triples])) / total) * 100 if total > 0 else 0


    dashboard_data = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M UTC"),
        "stats": {
            "novelty": round(novelty_score, 2),
            "uniqueness": round(uniqueness_score, 2),
            "total_generated": total
        },
        "hypotheses": decoded_hypotheses
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2)

    print(f"[SUCCESS] Dashboard JSON saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()