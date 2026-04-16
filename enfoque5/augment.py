"""Augmentación sintética: variaciones de SPA → reglas enfoque1 → MSLG sintético.

Uso:
    python -m enfoque5.augment

Genera enfoque5/data/augmented_train.txt con pares originales + sintéticos.
"""

import csv
import os
import random
import sys

import spacy

# Importar motor de reglas de enfoque1
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "enfoque1"))
from codigo import (
    aplicar_reglas_spa_a_mslg,
    construir_diccionario_compuestos,
    construir_diccionario_nombres,
)

from enfoque5 import config


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


# ── Augmentaciones sobre texto SPA ──

def word_dropout(texto, prob):
    """Elimina palabras al azar con probabilidad `prob`."""
    palabras = texto.split()
    if len(palabras) <= 2:
        return texto
    resultado = [w for w in palabras if random.random() > prob]
    return " ".join(resultado) if resultado else texto


def word_swap(texto, prob):
    """Intercambia pares de palabras adyacentes con probabilidad `prob`."""
    palabras = texto.split()
    for i in range(len(palabras) - 1):
        if random.random() < prob:
            palabras[i], palabras[i + 1] = palabras[i + 1], palabras[i]
    return " ".join(palabras)


def augmentar_spa(texto, n_variaciones, drop_prob, swap_prob):
    """Genera `n_variaciones` de una oración SPA."""
    variaciones = []
    for _ in range(n_variaciones):
        t = texto
        if random.random() < 0.5:
            t = word_dropout(t, drop_prob)
        if random.random() < 0.5:
            t = word_swap(t, swap_prob)
        if t != texto:
            variaciones.append(t)
    return variaciones


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    random.seed(config.RANDOM_SEED)

    print("Cargando dataset original...")
    pares = cargar_dataset(config.ORIGINAL_DATASET)
    train_pares, val_pares = split_train_val(
        pares, train_n=config.TRAIN_SPLIT, seed=config.RANDOM_SEED
    )
    print(f"  Original: {len(train_pares)} train | {len(val_pares)} val")

    print(f"Cargando spaCy ({config.SPACY_MODEL})...")
    nlp = spacy.load(config.SPACY_MODEL)

    print("Construyendo diccionarios de enfoque1...")
    dicc_compuestos = construir_diccionario_compuestos(train_pares)
    nombres_personas = construir_diccionario_nombres(train_pares)
    print(f"  {len(dicc_compuestos)} compuestos | {len(nombres_personas)} nombres")

    # ── Generar pares augmentados ──
    print(f"\nGenerando {config.AUGMENTATIONS_PER_SAMPLE} variaciones por muestra...")
    pares_augmentados = []
    aug_id = 0

    for par in train_pares:
        variaciones = augmentar_spa(
            par["spa"],
            config.AUGMENTATIONS_PER_SAMPLE,
            config.WORD_DROP_PROB,
            config.WORD_SWAP_PROB,
        )
        for var_spa in variaciones:
            mslg_sintetico = aplicar_reglas_spa_a_mslg(
                var_spa, nlp, dicc_compuestos, nombres_personas
            )
            if mslg_sintetico and len(mslg_sintetico.split()) >= 2:
                pares_augmentados.append({
                    "id": f"AUG{aug_id:04d}",
                    "mslg": mslg_sintetico,
                    "spa": var_spa,
                })
                aug_id += 1

    print(f"  Pares sintéticos generados: {len(pares_augmentados)}")

    # ── Combinar: originales (gold) + sintéticos (silver) ──
    todos = train_pares + pares_augmentados
    random.shuffle(todos)

    # ── Guardar dataset augmentado (train + val al final) ──
    output_path = config.AUGMENTED_DATASET
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("ID\tMSLG\tSPA\n")
        for par in todos:
            f.write(f"{par['id']}\t{par['mslg']}\t{par['spa']}\n")
        for par in val_pares:
            f.write(f"{par['id']}\t{par['mslg']}\t{par['spa']}\n")

    total_train = len(todos)
    print(f"\nDataset augmentado guardado: {output_path}")
    print(f"  Train: {total_train} ({len(train_pares)} gold + {len(pares_augmentados)} sintéticos)")
    print(f"  Val: {len(val_pares)} (gold, sin augmentar)")


if __name__ == "__main__":
    main()
