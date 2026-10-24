# Datasets

US road networks from the 9th DIMACS Implementation Challenge, series `USA-road-d` (weights =
distance), used in stages 1 and 3. NY, COL, CAL and E are versioned in this folder (88 MB in
total). The USA graph (351 MB) exceeds GitHub's limit of 100 MB per file, so it is ignored by git
(see `.gitignore`) and must be downloaded into this folder:

```bash
cd datasets
curl -fLO https://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.USA.gr.gz
gzip -t *.gz
python3 ../src/graphs/dimacs.py --check USA-road-d.USA.gr.gz
```

| File | Vertices | Arcs | .gz |
|---|---|---|---|
| USA-road-d.NY.gr.gz | 264,346 | 733,846 | 3.7 MB |
| USA-road-d.COL.gr.gz | 435,666 | 1,057,066 | 5.8 MB |
| USA-road-d.CAL.gr.gz | 1,890,815 | 4,657,742 | 27 MB |
| USA-road-d.E.gr.gz | 3,598,623 | 8,778,114 | 52 MB |
| USA-road-d.USA.gr.gz | 23,947,347 | 58,333,344 | 351 MB |

The synthetic graphs of stage 2 are not stored on disk: `src/benchmark.py --synthetic` generates them in
memory with a fixed seed (`src/graphs/graph.py`, `random_graph`).

`test/example.gr` is the 11-vertex example graph in DIMACS format, a
small instance for quickly testing the experimental routine (see "How to run" in the main README).
