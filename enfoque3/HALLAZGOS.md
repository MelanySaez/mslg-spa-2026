# Hallazgos — Pipeline SPA → MSLG (IberLEF 2026)

## Configuracion del Experimento

- **Modelo:** qwen2.5:14b (Ollama local)
- **Temperatura:** 0.1
- **Dataset:** 490 pares (400 pool + 90 validacion)
- **Metricas:** BLEU (corpus, smoothing 1), METEOR (promedio sentence), chrF (corpus)
- **Seed:** 42 (split reproducible)

---

## Resultados Comparativos

| Experimento  | BLEU   | METEOR | chrF   | Tiempo (s) |
|--------------|--------|--------|--------|------------|
| zero-shot    | 0.1292 | 0.4102 | 0.5833 | 441.3      |
| few-shot-3   | 0.0980 | 0.4304 | 0.5821 | 426.7      |
| few-shot-5   | 0.1035 | 0.4378 | 0.5897 | 417.6      |
| few-shot-10  | 0.1210 | 0.4533 | 0.6063 | 430.6      |
| few-shot-15  | 0.1220 | 0.4453 | 0.5979 | 440.1      |
| rag-5        | 0.1590 | 0.4992 | 0.6239 | 400.7      |
| rag-7        | 0.1346 | 0.5158 | 0.6323 | 392.3      |
| **rag-10**   | **0.2043** | **0.5117** | **0.6365** | 401.8 |

**Mejor configuracion global: RAG-10** (BLEU 0.2043, METEOR 0.5117, chrF 0.6365).

Tiempo total de ejecucion de los 8 experimentos: ~55.8 minutos (~6.2 min promedio por experimento, 90 oraciones cada uno).

---

## Hallazgos Clave

### 1. RAG supera consistentemente a Few-Shot y Zero-Shot

La recuperacion dinamica de ejemplos similares (RAG) produce mejores resultados que los ejemplos fijos en todas las metricas:

- **RAG-10 vs zero-shot:** +58% BLEU, +25% METEOR, +9% chrF
- **RAG-10 vs few-shot-15:** +67% BLEU, +15% METEOR, +6.5% chrF

Esto confirma que la relevancia de los ejemplos importa mas que la cantidad de ejemplos fijos. RAG le da al modelo pares de traduccion semanticamente cercanos a la oracion a traducir, lo que le permite inferir mejor los patrones especificos.

### 2. Few-Shot con pocos ejemplos puede empeorar respecto a Zero-Shot

Few-shot-3 obtuvo **peor BLEU (0.0980) que zero-shot (0.1292)**. Esto sugiere que pocos ejemplos fijos pueden sesgar al modelo hacia patrones particulares sin cubrir suficiente diversidad. El modelo empieza a imitar rigidamente los 3 ejemplos en lugar de aplicar las reglas generales del prompt.

METEOR y chrF si mejoran ligeramente con few-shot-3, lo que indica que el contenido lexico es mas cercano aunque el orden de las palabras empeora (BLEU penaliza fuertemente el orden).

### 3. Few-Shot satura alrededor de K=10

| K   | BLEU   | METEOR |
|-----|--------|--------|
| 3   | 0.0980 | 0.4304 |
| 5   | 0.1035 | 0.4378 |
| 10  | 0.1210 | 0.4533 |
| 15  | 0.1220 | 0.4453 |

De K=10 a K=15, BLEU apenas mejora (+0.8%) y METEOR baja (-1.8%). Mas ejemplos fijos no ayudan, probablemente porque el prompt se vuelve demasiado largo y el modelo pierde foco en la oracion a traducir.

### 4. Coincidencias exactas en RAG-10

De las 90 oraciones de validacion, RAG-10 logro **coincidencia exacta** en al menos 6 casos (~6.7%):

| SPA | Real | Prediccion |
|-----|------|------------|
| Tengo otro libro. | OTRO LIBRO YO TENER | OTRO LIBRO YO TENER |
| La sopa esta caliente. | SOPA CALIENTE | SOPA CALIENTE |
| Mi hermano revisa su agenda diario. | DIARIO MI HERMANO AGENDA REVISAR | DIARIO MI HERMANO AGENDA REVISAR |
| El ano pasado viaje a Guanajuato. | ANO-PASADO YO VIAJAR GUANAJUATO | ANO-PASADO YO VIAJAR GUANAJUATO |
| Yo abandone a mi hijo. | MI HIJO YO ABANDONAR | MI HIJO YO ABANDONAR |
| Hoy es mi cumpleanos. | HOY CUMPLEANOS MI | HOY CUMPLEANOS MI |

Estas coinciden en oraciones cortas y con estructura directa.

---

## Factores que Limitan los Resultados

### A. Orden de palabras en LSM

El factor mas impactante. La LSM tiene un orden sintactico propio (frecuentemente SOV o topicalizado) que no sigue reglas predecibles desde el espanol. Ejemplos:

| SPA | Real | Prediccion |
|-----|------|------------|
| Debes seguir estudiando | TU DEBER SEGUIR ESTUDIAR | ESTUDIAR TU DEBER |
| Mi tio vive en la frontera, en Tijuana | FRONTERA TIJUANA MI TIO VIVIR | MI TIO FRONTERA TIJUANA VIVIR |
| Ayer fui a la galeria | AYER YO YA IR GALERIA | AYER YO IR GALERIA |

El modelo capta los tokens correctos pero los ordena diferente. BLEU penaliza esto con fuerza.

### B. Vocabulario especifico de LSM

Algunas glosas usan conceptos que no se derivan directamente del espanol:

- "credencial del INE" → `CREDENCIAL VOTAR` (no `INE-CREDENCIAL`)
- "zoologico" → `ENTRAR ANIMAL` (concepto compuesto LSM, no la palabra "zoologico")
- "pareja" → `NOSOTROS-DE-DOS` (concepto relacional)
- "anoche" → `AYER NOCHE` (descomposicion temporal)

Estos requieren conocimiento especifico de LSM que el modelo no tiene.

### C. Marcadores aspectuales (YA, SIEMPRE)

La LSM usa `YA` como marcador de aspecto completivo y `SIEMPRE` como habitual. El modelo frecuentemente los omite o los coloca mal:

- Real: `AYER YO YA IR GALERIA` → Pred: `AYER YO IR GALERIA` (falta `YA`)
- Real: `DELEGACION YO YA IR ACTA LEVANTAR` → Pred: `DELEGACION YO IR ACTA LEVANTAR` (falta `YA`)

### D. Verbosidad del modelo

El LLM tiende a generar traducciones mas largas que la referencia, cuando el 76.5% de las glosas reales son mas cortas que el espanol:

- Real: `CUBA CALOR` (2 tokens) → Pred: `CUBA CLIMA CALOR MUCHO` (4 tokens)
- Real: `TODO AFORTUNADO` (2 tokens) → Pred: `TU AFORTUNADO VERDAD TU` (4 tokens)
- Real: `MI SOBRINO ALTO` (3 tokens) → Pred: `MI SOBRINO ALTO VERDAD` (4 tokens)

### E. Convencion dm- y conceptos compuestos

El modelo aplica `dm-` de forma inconsistente y no conoce los conceptos compuestos de LSM:

- `GUARDAR-BOLSILLO` (concepto unico en LSM) → el modelo pone `SU BOLSILO MONEDA GUARDAR`
- `dm-AIME` → el modelo pone `AIME` sin prefijo (o `DM-AIME` con mayusculas incorrectas)
- `BOLSA-MANO` → el modelo genera `BOLSA-DE-MANO` (agrega preposicion)

### F. Alucinaciones (principalmente en Zero-Shot)

Sin ejemplos de referencia, el modelo ocasionalmente genera tokens inexistentes o errores ortograficos:

- `QUIERETWEET` (contaminacion de datos de entrenamiento)
- `FEYO` en vez de `FEO`
- `PUDER` en vez de `PODER`
- `BOLSILO` en vez de `BOLSILLO`

RAG reduce significativamente estas alucinaciones al proveer contexto correcto.

---

## Conclusiones

1. **RAG es el enfoque ganador** para esta tarea con LLM sin entrenamiento. RAG-10 es la configuracion optima entre las probadas.

2. **El BLEU es bajo (0.20)** pero esperado para traduccion a una lengua de senas con orden libre y vocabulario especifico. chrF (0.64) refleja mejor la calidad real: a nivel de caracteres, las predicciones capturan buena parte del contenido.

3. **Los principales cuellos de botella** son el orden sintactico de LSM y el vocabulario especializado, ambos dificiles de capturar solo con in-context learning.

4. **Mejoras potenciales** para explorar:
   - Probar modelos mas grandes o multilingues (e.g. Llama 3.1 70B, Qwen2.5 72B)
   - Fine-tuning con LoRA sobre los 400 pares del pool
   - Agregar reglas de reordenamiento como post-procesamiento
   - Aumentar el dataset con back-translation o parafraseo
   - Combinar RAG + few-shot (ejemplos fijos + recuperados)

5. **Para la submission oficial** se recomienda usar la configuracion RAG-10 con `generate_submission.py`.
