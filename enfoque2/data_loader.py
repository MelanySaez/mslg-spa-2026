"""Carga y preprocesamiento del dataset MSLG_SPA para fine-tuning seq2seq."""

import csv
import random

from torch.utils.data import Dataset


def cargar_dataset(ruta_tsv: str):
    pares = []
    with open(ruta_tsv, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for fila in reader:
            pares.append({
                "id": fila["ID"].strip(),
                "mslg": fila["MSLG"].strip(),
                "spa": fila["SPA"].strip(),
            })
    return pares


def split_train_val(pares, train_n=400, seed=42):
    datos = pares[:]
    random.seed(seed)
    random.shuffle(datos)
    return datos[:train_n], datos[train_n:]


class MSLGDataset(Dataset):
    """Dataset seq2seq: input=SPA, target=MSLG."""

    def __init__(self, pares, tokenizer, max_source_len, max_target_len):
        self.pares = pares
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.pares)

    def __getitem__(self, idx):
        par = self.pares[idx]
        source = self.tokenizer(
            par["spa"],
            max_length=self.max_source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer(
            par["mslg"],
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": labels,
        }
