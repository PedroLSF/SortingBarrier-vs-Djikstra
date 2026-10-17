# Datasets

Malhas rodoviárias dos EUA do 9th DIMACS Implementation Challenge, série `USA-road-d` (pesos =
distância). Os arquivos não vão para o git (ver `.gitignore`); baixe-os para esta pasta:

```bash
cd datasets
for g in NY COL CAL E USA; do
  curl -fLO https://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.$g.gr.gz
done
gzip -t *.gz
python3 ../src/grafos/dimacs.py --check USA-road-d.NY.gr.gz
```

| Arquivo | Vértices | Arcos | .gz |
|---|---|---|---|
| USA-road-d.NY.gr.gz | 264.346 | 733.846 | 3,7 MB |
| USA-road-d.COL.gr.gz | 435.666 | 1.057.066 | 5,8 MB |
| USA-road-d.CAL.gr.gz | 1.890.815 | 4.657.742 | 27 MB |
| USA-road-d.E.gr.gz | 3.598.623 | 8.778.114 | 52 MB |
| USA-road-d.USA.gr.gz | 23.947.347 | 58.333.344 | 351 MB |

Os grafos sintéticos da etapa 2 não ficam em disco: o `src/benchmark.py --sintetico` os gera em
memória com seed fixa (`src/grafos/grafo.py`, `random_graph`).
