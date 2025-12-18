import torch
from torch.utils.data import Dataset

class DBLPDataset(Dataset):
    def __init__(self, triples_path):
        self.triples = []
        self.entities = set()
        self.relations = set()

        with open(triples_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    h, r, t = parts
                    self.triples.append((h, r, t))
                    self.entities.add(h)
                    self.entities.add(t)
                    self.relations.add(r)

        self.entity_list = sorted(list(self.entities))
        self.relation_list = sorted(list(self.relations))

        self.ent2id = {e: i for i, e in enumerate(self.entity_list)}
        self.rel2id = {r: i for i, r in enumerate(self.relation_list)}

        self.num_entities = len(self.entities)
        self.num_relations = len(self.relations)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        h, r, t = self.triples[idx]
        return {
            'head': torch.tensor(self.ent2id[h], dtype=torch.long),
            'relation': torch.tensor(self.rel2id[r], dtype=torch.long),
            'tail': torch.tensor(self.ent2id[t], dtype=torch.long)
        }