"""Evaluador para sentido reverso MSLG -> SPA.

Calcula BLEU/METEOR/chrF reusando enfoque3/evaluator.py (que asume las claves
'mslg_real' y 'mslg_pred', aquí mapeadas desde 'spa_real'/'spa_pred') y añade
COMET — métrica obligatoria del subtask MSLG2SPA según las bases de la tarea.

COMET se ejecuta vía el CLI 'comet-score' (instalado en un entorno aislado con
'uv tool install unbabel-comet') porque las versiones publicadas de
unbabel-comet pinnean pandas<3 y torch<2, incompatibles con el resto del
proyecto. Este wrapper escribe src/mt/ref a archivos temporales y parsea el
score del JSON de salida.

Si COMET no está disponible (binario faltante, modelo no descargable, etc.),
se reporta None y se continúa con BLEU/METEOR/chrF: la actividad evaluará
COMET sobre el .txt enviado en cualquier caso.
"""

import importlib.util
import json
import os
import shutil
import subprocess
import tempfile

import config

_path = os.path.join(os.path.dirname(__file__), "..", "..", "enfoque3",
                     "evaluator.py")
_spec = importlib.util.spec_from_file_location("enfoque3_evaluator", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_evaluate_inner = _mod.evaluate


def _extract_system_score(data) -> float | None:
    """Extrae el system score de COMET de cualquier estructura JSON razonable.

    El CLI 'comet-score --to_json' (versión 2.x) emite:
        {"<ruta_hyp>": [{"src": ..., "mt": ..., "ref": ..., "COMET": <float>}, ...]}
    sin un campo 'system_score' global. Se promedia el campo por segmento.
    """
    if isinstance(data, dict):
        # Campos directos comunes
        for key in ("system_score", "system-score", "score", "mean", "avg"):
            if key in data:
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    pass
        # Si los values son listas de segmentos (formato comet-score 2.x), promediar
        # los scores de cada segmento sobre todas las listas.
        per_segment = []
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                for seg in v:
                    for k in ("COMET", "comet", "score", "system_score"):
                        if k in seg and isinstance(seg[k], (int, float)):
                            per_segment.append(float(seg[k]))
                            break
        if per_segment:
            return sum(per_segment) / len(per_segment)
        # Recursión sobre valores anidados
        for v in data.values():
            nested = _extract_system_score(v)
            if nested is not None:
                return nested
        # Promedio si hay lista de scores nombrada
        for key in ("scores", "segment_scores", "sentence_scores"):
            if key in data and isinstance(data[key], list):
                nums = [float(x) for x in data[key]
                        if isinstance(x, (int, float))]
                if nums:
                    return sum(nums) / len(nums)
    elif isinstance(data, list):
        if data and isinstance(data[0], dict):
            # Lista de segmentos con clave 'COMET' (o variantes)
            scores = []
            for seg in data:
                for k in ("COMET", "comet", "score", "system_score"):
                    if k in seg and isinstance(seg[k], (int, float)):
                        scores.append(float(seg[k]))
                        break
            if scores:
                return sum(scores) / len(scores)
            for item in data:
                nested = _extract_system_score(item)
                if nested is not None:
                    return nested
        if data and all(isinstance(x, (int, float)) for x in data):
            return sum(data) / len(data)
    return None


def _run_comet_python_api(srcs: list, hyps: list, refs: list) -> float | None:
    """Fallback: usa la API Python de comet directamente (sin CLI)."""
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        return None
    try:
        model_path = download_model(config.COMET_MODEL)
        model = load_from_checkpoint(model_path)
        data = [{"src": s, "mt": h, "ref": r}
                for s, h, r in zip(srcs, hyps, refs)]
        gpus = config.COMET_GPUS
        output = model.predict(data, batch_size=config.COMET_BATCH_SIZE,
                               gpus=gpus)
        score = getattr(output, "system_score", None)
        if score is None and hasattr(output, "scores"):
            scores = output.scores
            if scores:
                score = sum(scores) / len(scores)
        return float(score) if score is not None else None
    except Exception as exc:
        print(f"WARNING: API Python COMET falló: {exc}")
        return None


def _run_comet(srcs: list, hyps: list, refs: list) -> float | None:
    """Invoca 'comet-score' vía subprocess y retorna el system score."""
    binary = shutil.which(config.COMET_BIN) or config.COMET_BIN
    if not shutil.which(config.COMET_BIN):
        # Intentar aun así (ej. ruta absoluta en COMET_BIN o ya en PATH).
        if not os.path.exists(binary):
            print(f"WARNING: '{config.COMET_BIN}' no encontrado en PATH. "
                  f"Intentando API Python de comet...")
            return _run_comet_python_api(srcs, hyps, refs)

    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, "src.txt")
        hyp_path = os.path.join(tmp, "hyp.txt")
        ref_path = os.path.join(tmp, "ref.txt")
        json_path = os.path.join(tmp, "out.json")

        for path, lines in (
            (src_path, srcs),
            (hyp_path, hyps),
            (ref_path, refs),
        ):
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for line in lines:
                    f.write(line.replace("\n", " ").strip() + "\n")

        cmd = [
            binary,
            "-s", src_path,
            "-t", hyp_path,
            "-r", ref_path,
            "--model", config.COMET_MODEL,
            "--batch_size", str(config.COMET_BATCH_SIZE),
            "--gpus", str(config.COMET_GPUS),
            "--quiet",
            "--to_json", json_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=1800)
        except FileNotFoundError:
            print(f"WARNING: no se pudo ejecutar '{binary}'. Saltando COMET.")
            return None
        except subprocess.TimeoutExpired:
            print("WARNING: COMET excedió el timeout (30 min). Saltando.")
            return None

        if result.returncode != 0:
            print(f"WARNING: COMET retornó código {result.returncode}.")
            if result.stderr:
                print(result.stderr.strip().splitlines()[-3:])
            return None

        # Si hay JSON, intentar parsearlo
        score = None
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            score = _extract_system_score(data)
            if score is None:
                # Persistir JSON crudo para diagnóstico
                debug_dst = os.path.join(config.RESULTS_DIR, "_comet_raw.json")
                os.makedirs(config.RESULTS_DIR, exist_ok=True)
                try:
                    with open(debug_dst, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    print(f"WARNING: estructura JSON COMET inesperada. "
                          f"Volcado para diagnóstico: {debug_dst}")
                except Exception:
                    pass
                if isinstance(data, dict):
                    print(f"  keys top-level: {list(data.keys())[:10]}")

        # Fallback: parsear stdout
        if score is None:
            stdout_lines = (result.stdout or "").strip().splitlines()
            for line in reversed(stdout_lines):
                low = line.lower()
                if "system score" in low or low.startswith("score:"):
                    parts = line.replace("=", ":").split(":")
                    try:
                        return float(parts[-1].strip())
                    except ValueError:
                        continue
            print("WARNING: COMET stdout sin score parseable.")
            if stdout_lines:
                print(f"  últimas líneas stdout: {stdout_lines[-3:]}")

        return score


def evaluate(results):
    """Calcula BLEU, METEOR, chrF y (opcional) COMET.

    Args:
        results: lista de dicts con claves 'mslg', 'spa_real', 'spa_pred'.

    Returns:
        dict con 'bleu', 'meteor', 'chrf' y 'comet' (None si no disponible).
    """
    adapted = [
        {"mslg_real": r["spa_real"], "mslg_pred": r["spa_pred"]}
        for r in results
    ]
    metrics = _evaluate_inner(adapted)

    comet_score = None
    if getattr(config, "ENABLE_COMET", True):
        srcs = [r["mslg"] for r in results]
        hyps = [r["spa_pred"] for r in results]
        refs = [r["spa_real"] for r in results]
        print(f"  Ejecutando COMET ({config.COMET_MODEL}) sobre "
              f"{len(results)} pares — esto puede tardar...")
        comet_score = _run_comet(srcs, hyps, refs)
        if comet_score is not None:
            print(f"  COMET system score: {comet_score:.4f}")

    metrics["comet"] = comet_score
    return metrics
