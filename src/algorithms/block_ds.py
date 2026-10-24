"""
Block-based data structure D of Lemma 3.3 in Duan et al. (2025), used by DuanSSSP.bmssp.

It replaces Dijkstra's priority queue: instead of extracting one minimum at a time it hands
out batches of the M smallest keys (Pull), and accepts batches of keys smaller than all its
contents (BatchPrepend). This is what lets BMSSP avoid sorting every vertex by distance.

Bounds (Lemma 3.3), for N stored pairs and block size M:
    Insert          O(max{1, log(N/M)}) amortized
    BatchPrepend    O(L · max{1, log(L/M)}) for L pairs
    Pull            O(M) amortized
"""

import random
from bisect import bisect_left
from collections import deque
from itertools import compress
from operator import not_

from .common import KEY_INF, TRACE, CountedKey, log


class _Block:
    """Block of the structure D: items maps key -> value; upper is the block's upper bound
    (only blocks of D1 have one; blocks of D0 keep None)."""
    __slots__ = ("items", "upper")

    def __init__(self, items=None, upper=None):
        self.items = items if items is not None else {}
        self.upper = upper


def _counted_bisect(a, x):
    """bisect.bisect_left(a, x) that adds every comparison it makes to CountedKey.n."""
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) >> 1
        CountedKey.n += 1
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


_select_rng = random.Random(2025)


def _partition(a, p, count):
    """Split the (key, value) pairs of a into (value < p, value > p), one comparison each.

    Values are distinct (the total order of Assumption 2.1) and p itself is not in a.
    With count=True the len(a) comparisons are added to CountedKey.n.
    """
    flags = [kv[1] < p for kv in a]
    if count:
        CountedKey.n += len(a)
    return list(compress(a, flags)), list(compress(a, map(not_, flags)))


def _select(a, k, count):
    """Select the k pairs of smallest value in a, for 0 < k < len(a).

    Returns (k smallest, the rest, largest value among the k smallest); a is consumed.
    Lemma 3.3 needs linear-time selection: this is quickselect, which partitions around a
    random pivot (seeded _select_rng, so runs are reproducible) and recurses only into the
    side holding the k-th element, for O(len(a)) expected comparisons. The paper cites the
    deterministic selection of [BFP+73] instead; only the guarantee changes, from worst
    case to expected.
    """
    smaller, larger = [], []
    while True:
        if len(a) == 1:
            smaller += a
            return smaller, larger, a[0][1]
        i = _select_rng.randrange(len(a))
        piv = a[i]
        a[i] = a[-1]
        a.pop()
        lo, hi = _partition(a, piv[1], count)
        if k <= len(lo):
            larger += hi
            larger.append(piv)
            a = lo
        elif k == len(lo) + 1:
            smaller += lo
            smaller.append(piv)
            larger += hi
            return smaller, larger, piv[1]
        else:
            smaller += lo
            smaller.append(piv)
            k -= len(lo) + 1
            a = hi


class BlockDS:
    """Block-based structure D of Lemma 3.3, holding (key, value) pairs with values < B.

    Blocks live in two sequences:
        D0: blocks created by BatchPrepend (any number of them), in a deque;
        D1: blocks created by Insert, each with an upper bound, kept sorted by bound.
    Every block holds at most M pairs; `where` maps each key to its block, so Delete is O(1).

    Differences from the paper:
        * the red-black tree over the D1 bounds is replaced by a sorted list `uppers`
          searched with bisect (same search, same O(log) cost per lookup);
        * Split, Pull and BatchPrepend find the k smallest by quickselect (_select),
          without sorting;
        * emptied blocks are not removed right away; empty blocks at the front of each
          sequence are dropped at the next Pull (_compact), as the paper does by counting
          deleted elements in the block size.

    With count=True every comparison executed inside D is added to CountedKey.n.
    fmt and depth are only used for TRACE logs.
    """

    def __init__(self, M, B, fmt, depth, count=False):
        self.M = max(1, M)
        self.B = B
        self.D0 = deque()
        self.D1 = [_Block(upper=B)]
        self.uppers = [B]
        self.where = {}
        self.fmt = fmt
        self.depth = depth
        self.n_insert = 0
        self.n_prepend = 0
        self.tracing = log.isEnabledFor(TRACE)
        self.count = count
        self._bisect = _counted_bisect if count else bisect_left

    def __len__(self):
        """Number of keys currently stored."""
        return len(self.where)

    def _trace(self, msg, *args):
        """Log an internal operation of D at the TRACE level, indented by recursion depth."""
        log.log(TRACE, "%s    [D] " + msg, "│   " * self.depth, *args)

    def _delete(self, key):
        """Delete: remove key from its block in O(1), leaving the block in place."""
        blk = self.where.pop(key)
        del blk.items[key]

    def insert(self, key, value):
        """Insert(key, value), amortized O(max{1, log(N/M)}).

        If key is already present with a value <= value, nothing changes; otherwise the old
        pair is deleted. The pair goes into the first D1 block whose upper bound is >= value,
        and a block that grows past M is split. Returns True if the pair was stored.
        """
        old = self.where.get(key)
        if old is not None:
            if old.items[key] <= value:
                return False
            self._delete(key)
        i = min(self._bisect(self.uppers, value), len(self.D1) - 1)
        if self.count:
            CountedKey.n += old is not None
        blk = self.D1[i]
        blk.items[key] = value
        self.where[key] = blk
        self.n_insert += 1
        if self.tracing:
            self._trace("Insert(%s, %s) -> block D1[%d]", key, self.fmt(value), i)
        if len(blk.items) > self.M:
            self._split(i)
        return True

    def _split(self, i):
        """Split block D1[i] at its median value into two blocks; the left one takes the
        median as its upper bound, the right one keeps the old bound."""
        blk = self.D1[i]
        items = list(blk.items.items())
        left_items, right_items, bound = _select(items, len(items) // 2, self.count)
        left = _Block(dict(left_items), upper=bound)
        right = _Block(dict(right_items), upper=blk.upper)
        for b in (left, right):
            for k in b.items:
                self.where[k] = b
        self.D1[i:i + 1] = [left, right]
        self.uppers[i:i + 1] = [left.upper, right.upper]
        if self.tracing:
            self._trace("Split of D1[%d]: %d + %d items", i, len(left.items), len(right.items))

    def batch_prepend(self, pairs):
        """BatchPrepend(pairs): add L pairs whose values are all smaller than every value in D.

        Duplicate keys keep their smallest value, and keys already in D with a value that is
        not larger are skipped. The surviving pairs become one block if L <= M, otherwise
        about L/M blocks built by successive medians, placed at the front of D0.
        Returns the number of pairs stored.
        """
        best = {}
        compared = 0
        for k, v in pairs:
            b = best.get(k)
            if b is None:
                best[k] = v
            else:
                compared += 1
                if v < b:
                    best[k] = v
        items = []
        for k, v in best.items():
            old = self.where.get(k)
            if old is not None:
                compared += 1
                if old.items[k] <= v:
                    continue
                self._delete(k)
            items.append((k, v))
        if self.count:
            CountedKey.n += compared
        if not items:
            return 0
        if len(items) <= self.M:
            blocks = [_Block(dict(items))]
        else:
            blocks = [_Block(dict(part)) for part in self._into_blocks(items)]
        for b in blocks:
            for k in b.items:
                self.where[k] = b
        self.D0.extendleft(reversed(blocks))
        self.n_prepend += len(items)
        if self.tracing:
            self._trace("BatchPrepend of %d pairs into %d block(s) of D0", len(items), len(blocks))
        return len(items)

    def _into_blocks(self, items):
        """Split items into blocks of at most M pairs, ordered among themselves, by successive
        medians: O(L log(L/M)) comparisons for L items."""
        if len(items) <= self.M:
            return [items]
        left_items, right_items, _ = _select(items, len(items) // 2, self.count)
        return self._into_blocks(left_items) + self._into_blocks(right_items)

    def _compact(self):
        """Drop the empty blocks at the front of D0 and D1. The last D1 block, whose upper
        bound is B, is always kept so that Insert always has a target block."""
        D0 = self.D0
        while D0 and not D0[0].items:
            D0.popleft()
        h = 0
        while h < len(self.D1) - 1 and not self.D1[h].items:
            h += 1
        if h:
            del self.D1[:h]
            del self.uppers[:h]

    def pull(self):
        """Pull(): remove and return the (up to) M keys of smallest value.

        Candidates are gathered from the first blocks of D0 and D1 until each sequence
        contributes at least M pairs, and the M smallest are selected. The smallest
        remaining value sits in the first non-empty block of D0 or of D1.

        Returns (x, S): S is the list of keys removed and x is a bound separating them from
        what is left (the smallest remaining value, or B if D became empty).
        """
        self._compact()
        M = self.M
        cand = []
        for seq in (self.D0, self.D1):
            got = 0
            for blk in seq:
                if got >= M:
                    break
                cand.extend(blk.items.items())
                got += len(blk.items)
        if len(cand) > M:
            cand, _, _ = _select(cand, M, self.count)
        for k, _ in cand:
            self._delete(k)

        if not self.where:
            x = self.B
        else:
            self._compact()
            b0 = self.D0[0].items if self.D0 else None
            b1 = next((b.items for b in self.D1 if b.items), None)
            first_d0 = min(b0.values()) if b0 else KEY_INF
            first_d1 = min(b1.values()) if b1 else KEY_INF
            x = min(first_d0, first_d1)
            if self.count:
                CountedKey.n += (len(b0) - 1 if b0 else 0) + (len(b1) - 1 if b1 else 0) + 1
        S = [k for k, _ in cand]
        if self.tracing:
            self._trace("Pull -> %d key(s), x = %s", len(S), self.fmt(x))
        return x, S
