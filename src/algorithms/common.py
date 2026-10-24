"""
Shared infrastructure of the two shortest-path algorithms (duan.py and dijkstra.py).

Everything here must be one single object seen by both algorithms, so that a benchmark run
switches or counts both in the same way: the infinity constants, the logger with its extra
TRACE level, the selectable binary heap (use_heap / heap_ops), the comparison-counting
key (CountedKey) used by the counting runs, and the check that both return the same
distances (mismatches).
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


def use_heap(which):
    """Select the binary heap used by dijkstra() and DuanSSSP.base_case().

    which: 'c' (default) uses the C-accelerated heapq; 'python' loads the standard library's
    heapq.py with the _heapq C module blocked, so the same algorithm runs in pure Python.
    Comparing both isolates the interpreter's share of the running time.
    """
    global heappush, heappop, _heapq_py
    if which == "python" and _heapq_py is None:
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
    mod = _heapq_py if which == "python" else heapq
    heappush, heappop = mod.heappush, mod.heappop


class CountedKey:
    """Wrapper key for counting runs: every `<` it evaluates is one comparison of the model.

    The class attribute CountedKey.n is a global counter. It accumulates the exact number of
    comparisons made inside the heap (whose items are wrapped in CountedKey) and inside the
    structure D (through _counted_bisect and _partition).
    """
    __slots__ = ("pair",)
    n = 0

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, o):
        CountedKey.n += 1
        return self.pair < o.pair


def heap_ops(count):
    """Return the (push, pop) pair used by dijkstra() and DuanSSSP.base_case().

    With count=False they are the heap functions currently selected by use_heap().
    With count=True every item is wrapped in a CountedKey on push and unwrapped on pop, so
    each `<` the heap evaluates is counted in CountedKey.n. Call it at the start of every run,
    so that a later use_heap() takes effect.
    """
    push, pop = heappush, heappop
    if not count:
        return push, pop
    return (lambda heap, item: push(heap, CountedKey(item))), (lambda heap: pop(heap).pair)


def mismatches(a, b):
    """Vertices whose distances in a and b differ beyond a relative tolerance of 1e-9.

    Used by bmssp.py and benchmark.py to check DuanSSSP against Dijkstra. Integer weights
    give identical lists, so the exact test a == b settles almost every call quickly.
    """
    if a == b:
        return []
    return [v for v, (x, y) in enumerate(zip(a, b)) if not (x == y or math.isclose(x, y, rel_tol=1e-9))]
