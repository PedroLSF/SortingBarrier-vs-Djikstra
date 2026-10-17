"""
Shared infrastructure of the two shortest-path algorithms (duan.py and dijkstra.py).

Everything here must be one single object seen by both algorithms, so that a benchmark run
switches or counts both in the same way: the infinity constants, the logger with its extra
TRACE level, the selectable binary heap (usar_heap / heap_ops) and the comparison-counting
key (Chave) used by the counting runs.
"""

import heapq
import logging
import math
import sys

INF = math.inf
KEY_INF = (INF, INF, INF)

TRACE = 5
logging.addLevelName(TRACE, "TRACE")
log = logging.getLogger("bmssp")

heappush, heappop = heapq.heappush, heapq.heappop
_heapq_py = None


def usar_heap(qual):
    """Select the binary heap used by dijkstra() and DuanSSSP.base_case().

    qual: 'c' (default) uses the C-accelerated heapq; 'python' loads the standard library's
    heapq.py with the _heapq C module blocked, so the same algorithm runs in pure Python.
    Comparing both isolates the interpreter's share of the running time.
    """
    global heappush, heappop, _heapq_py
    if qual == "python" and _heapq_py is None:
        import importlib.util
        spec = importlib.util.find_spec("heapq")
        _heapq_py = importlib.util.module_from_spec(spec)
        c = sys.modules.get("_heapq")
        sys.modules["_heapq"] = None
        try:
            spec.loader.exec_module(_heapq_py)
        finally:
            if c is None:
                del sys.modules["_heapq"]
            else:
                sys.modules["_heapq"] = c
    mod = _heapq_py if qual == "python" else heapq
    heappush, heappop = mod.heappush, mod.heappop


class Chave:
    """Wrapper key for counting runs: every `<` it evaluates is one comparison of the model.

    The class attribute Chave.n is a global counter. It accumulates the exact number of
    comparisons made inside the heap (whose items are wrapped in Chave) and inside the
    structure D (through _bisect_contado and _particiona).
    """
    __slots__ = ("par",)
    n = 0

    def __init__(self, par):
        self.par = par

    def __lt__(self, o):
        Chave.n += 1
        return self.par < o.par


def heap_ops(contar):
    """Return the (push, pop) pair used by dijkstra() and DuanSSSP.base_case().

    With contar=False they are the heap functions currently selected by usar_heap().
    With contar=True every item is wrapped in a Chave on push and unwrapped on pop, so each
    `<` the heap evaluates is counted in Chave.n. Call it at the start of every run, so that
    a later usar_heap() takes effect.
    """
    push, pop = heappush, heappop
    if not contar:
        return push, pop
    return (lambda heap, item: push(heap, Chave(item))), (lambda heap: pop(heap).par)
