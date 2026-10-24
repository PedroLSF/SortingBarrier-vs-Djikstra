# Datasets

US road networks from the 9th DIMACS Implementation Challenge, series `USA-road-d` (weights =
distance). The files do not go into git (see `.gitignore`); download them into this folder:

```bash
cd datasets
for g in NY COL CAL E USA; do
  curl -fLO https://www.diag.uniroma1.it/challenge9/data/USA-road-d/USA-road-d.$g.gr.gz
done
gzip -t *.gz
python3 ../src/graphs/dimacs.py --check USA-road-d.NY.gr.gz
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

`test/example.gr` is the only graph versioned in git: the 11-vertex example graph in DIMACS format, a
small instance for quickly testing the experimental routine (see "How to run" in the main README).
