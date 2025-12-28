import pandas as pd
import torch
import json
import plotly.express as px
from pathlib import Path
import os
import sys
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

CHECKPOINT_PATH = Path(parent_dir) / "checkpoints/gan_latest.pth"
MAPPINGS_FILE = Path(parent_dir) / "data/processed/kg_mappings.json"
OUTPUT_HTML = Path(parent_dir) / "semantic_map.html"

MAX_NODES = 1000

def create_interactive_map():
    if not CHECKPOINT_PATH.exists():
        return

    with open(MAPPINGS_FILE, "r") as f:
        data = json.load(f)
        id_to_str = data.get("id2ent", {})

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    if "D_state" in checkpoint:
        embeddings = checkpoint["D_state"]["ent_embedding.weight"].numpy()
    else:
        return

    indices = []
    metadata = []
    
    sorted_ids = sorted([int(k) for k in id_to_str.keys()])
    step = max(1, len(sorted_ids) // MAX_NODES)
    
    for i in range(0, len(sorted_ids), step):
        idx = sorted_ids[i]
        if idx >= len(embeddings): continue
        
        indices.append(idx)
        text = id_to_str[str(idx)]
        
        if text.startswith("venue/"): e_type = "Venue"
        elif text.startswith("conf/"): e_type = "Conference Paper"
        elif text.startswith("journals/"): e_type = "Journal Article"
        elif text.startswith("homepages/"): e_type = "Person Profile"
        elif " " in text and not text.startswith("dblp:"): e_type = "Author/Person" 
        elif text.startswith("dblp:"): e_type = "Relation/Type"
        else: e_type = "Other"
        
        display_name = (text[:75] + '..') if len(text) > 75 else text
        metadata.append({"Name": display_name, "Type": e_type})

    if not indices:
        return

    raw_embeddings = embeddings[indices]
    selected_embeddings = normalize(raw_embeddings, norm='l2')
    
    tsne = TSNE(
        n_components=2, 
        perplexity=30,
        n_iter=2000,
        random_state=42, 
        init='random',
        learning_rate='auto'
    )
    coords = tsne.fit_transform(selected_embeddings)

    df = pd.DataFrame(metadata)
    df["X"] = coords[:, 0]
    df["Y"] = coords[:, 1]

    fig = px.scatter(
        df, x="X", y="Y", color="Type", hover_name="Name",
        template="plotly_dark",
        title=f"Latent Space Manifold (t-SNE Projection - {len(df)} Nodes)",
        color_discrete_map={
            "Venue": "#00ffff",       
            "Author/Person": "#ff0000", 
            "Conference Paper": "#00ff00",
            "Journal Article": "#ffff00",  
            "Person Profile": "#ff00ff",   
            "Other": "#808080"        
        }
    )

    fig.update_traces(marker=dict(size=7, opacity=0.8, line=dict(width=0.5, color='white')))
    fig.update_layout(xaxis_title="t-SNE Dimension 1", yaxis_title="t-SNE Dimension 2")
    fig.write_html(str(OUTPUT_HTML))

if __name__ == "__main__":
    create_interactive_map()