#!/usr/bin/env python3
"""
Implementação didática, com logs, do algoritmo de

    R. Duan, J. Mao, X. Mao, X. Shu, L. Yin.
    "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths".
    arXiv:2504.17033v2, jul/2025.  https://arxiv.org/abs/2504.17033

O artigo apresenta um SSSP determinístico em O(m log^(2/3) n) no modelo
comparação-adição, superando o O(m + n log n) de Dijkstra em grafos esparsos.

Mapeamento código <-> artigo:
    find_pivots()      -> Algoritmo 1 (Lemma 3.2)
    base_case()        -> Algoritmo 2
    bmssp()            -> Algoritmo 3 (Lemma 3.1)
    BlockDS            -> estrutura de dados do Lemma 3.3 (Insert / BatchPrepend / Pull)
    key() / _can_relax -> Assumption 2.1 (ordem total entre caminhos)
    to_constant_degree -> transformação para grau constante (Seção 2)

Observação: o ganho do algoritmo é assintótico e vale no modelo comparação-adição;
em tempo de parede, em Python, o Dijkstra com heapq é mais rápido nos grafos testados.
Os logs servem para acompanhar a execução; a medição fica em benchmark.py.
"""

import argparse
import heapq
import logging
import math
import random
import sys
import time
from bisect import bisect_left
from collections import deque
from operator import itemgetter

from dimacs import load_gr

INF = math.inf
KEY_INF = (INF, INF, INF)  # cota B = ∞ na ordem total (d, hops, vértice)

TRACE = 5  # nível abaixo de DEBUG para as operações internas da estrutura D
logging.addLevelName(TRACE, "TRACE")
log = logging.getLogger("bmssp")


# --------------------------------------------------------------------------- #
# Grafo
# --------------------------------------------------------------------------- #
class Graph:
    """adj[u] é uma sequência intercalada (v0, w0, v1, w1, ...): 16 B por arco com
    ints compartilhados, contra ~160 B em listas de tuplas; o USA (58M arcos)
    só cabe em memória assim. Iterar com it = iter(adj[u]); zip(it, it)."""

    def __init__(self, n, names=None, adj=None):
        self.n = n
        self.adj = adj if adj is not None else [[] for _ in range(n)]
        self.m = sum(len(a) for a in adj) // 2 if adj is not None else 0
        self.names = names

    def add_edge(self, u, v, w):
        if w < 0:
            raise ValueError("pesos devem ser não negativos")
        self.adj[u] += (v, w)
        self.m += 1

    def arcs(self, u):
        it = iter(self.adj[u])
        return zip(it, it)

    def name(self, v):
        return self.names[v] if self.names else str(v)


def to_constant_degree(g, source):
    """Seção 2 ("Constant-Degree Graph"): cada vértice v vira um ciclo de peso 0
    com um nó x_vw por aresta incidente; a aresta (u,v) vira x_uv -> x_vu.
    Grau de entrada e saída <= 2. Retorna (G', s', rep) onde rep[v] é um nó do
    ciclo de v (todos têm a mesma distância)."""
    slots = [[] for _ in range(g.n)]
    owner = []
    edges = []

    def new_node(v):
        owner.append(v)
        slots[v].append(len(owner) - 1)
        return len(owner) - 1

    for u in range(g.n):
        for v, w in g.arcs(u):
            edges.append((new_node(u), new_node(v), w))
    for v in range(g.n):
        if not slots[v]:
            new_node(v)

    names = [f"{g.name(owner[x])}#{slots[owner[x]].index(x)}" for x in range(len(owner))]
    g2 = Graph(len(owner), names)
    for v in range(g.n):
        cyc = slots[v]
        if len(cyc) > 1:
            for i, x in enumerate(cyc):
                g2.add_edge(x, cyc[(i + 1) % len(cyc)], 0)
    for a, b, w in edges:
        g2.add_edge(a, b, w)
    rep = [slots[v][0] for v in range(g.n)]
    return g2, rep[source], rep


# --------------------------------------------------------------------------- #
# Estrutura de dados do Lemma 3.3
# --------------------------------------------------------------------------- #
class _Block:
    __slots__ = ("items", "upper")

    def __init__(self, items=None, upper=None):
        self.items = items if items is not None else {}  # chave -> valor
        self.upper = upper  # cota superior (apenas blocos de D1)


_VALUE = itemgetter(1)


class BlockDS:
    """Lista de blocos em duas sequências:
       D0: blocos vindos de BatchPrepend (sem limite de quantidade);
       D1: blocos vindos de Insert, cada um com cota superior, ordenados.
    Cada bloco tem no máximo M pares. A árvore rubro-negra de cotas do artigo
    é substituída por uma lista ordenada + bisect (mesma ideia de busca).
    Blocos esvaziados não são removidos na hora: Delete fica O(1) e os vazios
    na frente de cada sequência são descartados no Pull seguinte, como o
    artigo faz ao contar os elementos já deletados no tamanho do bloco."""

    def __init__(self, M, B, fmt, depth, contar=False):
        self.M = max(1, M)
        self.B = B
        self.D0 = deque()
        self.D1 = [_Block(upper=B)]
        self.uppers = [B]
        self.where = {}  # chave -> bloco que a contém
        self.fmt = fmt
        self.depth = depth
        self.n_insert = 0
        self.n_prepend = 0
        self.tracing = log.isEnabledFor(TRACE)
        self.contar = contar
        self.comparacoes = 0  # estimativa das comparações internas (bisect, sort, min), só com contar=True

    def __len__(self):
        return len(self.where)

    def _trace(self, msg, *args):
        log.log(TRACE, "%s    [D] " + msg, _indent(self.depth), *args)

    def _delete(self, key):
        blk = self.where.pop(key)
        del blk.items[key]

    def insert(self, key, value):
        """Insert: amortizado O(max{1, log(N/M)})."""
        old = self.where.get(key)
        if old is not None:
            if old.items[key] <= value:
                return False
            self._delete(key)
        i = min(bisect_left(self.uppers, value), len(self.D1) - 1)
        if self.contar:
            self.comparacoes += len(self.uppers).bit_length() + (old is not None)
        blk = self.D1[i]
        blk.items[key] = value
        self.where[key] = blk
        self.n_insert += 1
        if self.tracing:
            self._trace("Insert(%s, %s) -> bloco D1[%d]", key, self.fmt(value), i)
        if len(blk.items) > self.M:
            self._split(i)
        return True

    def _split(self, i):
        # O artigo usa mediana em O(M) [BFP+73]; em Python o sort em C sai mais barato.
        blk = self.D1[i]
        ordered = sorted(blk.items.items(), key=_VALUE)
        if self.contar:
            self.comparacoes += len(ordered) * len(ordered).bit_length()
        half = len(ordered) // 2
        left = _Block(dict(ordered[:half]), upper=ordered[half - 1][1])
        right = _Block(dict(ordered[half:]), upper=blk.upper)
        for b in (left, right):
            for k in b.items:
                self.where[k] = b
        self.D1[i:i + 1] = [left, right]
        self.uppers[i:i + 1] = [left.upper, right.upper]
        if self.tracing:
            self._trace("Split de D1[%d]: %d + %d elementos", i, len(left.items), len(right.items))

    def batch_prepend(self, pairs):
        """BatchPrepend: todos os valores são menores que qualquer valor em D."""
        best = {}
        for k, v in pairs:
            if k not in best or v < best[k]:
                best[k] = v
        items = []
        for k, v in best.items():
            old = self.where.get(k)
            if old is not None:
                if old.items[k] <= v:
                    continue
                self._delete(k)
            items.append((k, v))
        if self.contar:
            self.comparacoes += len(pairs) + len(items)
        if not items:
            return 0
        if len(items) <= self.M:
            blocks = [_Block(dict(items))]
        else:
            items.sort(key=_VALUE)
            if self.contar:
                self.comparacoes += len(items) * len(items).bit_length()
            size = max(1, (self.M + 1) // 2)
            blocks = [_Block(dict(items[j:j + size])) for j in range(0, len(items), size)]
        for b in blocks:
            for k in b.items:
                self.where[k] = b
        self.D0.extendleft(reversed(blocks))
        self.n_prepend += len(items)
        if self.tracing:
            self._trace("BatchPrepend de %d pares em %d bloco(s) de D0", len(items), len(blocks))
        return len(items)

    def _compact(self):
        D0 = self.D0
        while D0 and not D0[0].items:
            D0.popleft()
        h = 0
        while h < len(self.D1) - 1 and not self.D1[h].items:  # o último bloco (cota B) fica
            h += 1
        if h:
            del self.D1[:h]
            del self.uppers[:h]

    def pull(self):
        """Pull: devolve até M chaves de menor valor e uma cota x que as separa
        do restante (x = B se D ficar vazia)."""
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
            cand.sort(key=_VALUE)
            if self.contar:
                self.comparacoes += len(cand) * len(cand).bit_length()
            del cand[M:]
        for k, _ in cand:
            self._delete(k)

        if not self.where:
            x = self.B
        else:
            self._compact()
            # o menor restante está no primeiro bloco não vazio de D0 ou de D1
            b0 = self.D0[0].items if self.D0 else None
            b1 = next((b.items for b in self.D1 if b.items), None)
            first_d0 = min(b0.values()) if b0 else KEY_INF
            first_d1 = min(b1.values()) if b1 else KEY_INF
            x = min(first_d0, first_d1)
            if self.contar:
                self.comparacoes += (len(b0) if b0 else 0) + (len(b1) if b1 else 0)
        S = [k for k, _ in cand]
        if self.tracing:
            self._trace("Pull -> %d chave(s), x = %s", len(S), self.fmt(x))
        return x, S


# --------------------------------------------------------------------------- #
# Algoritmo principal
# --------------------------------------------------------------------------- #
def _indent(depth):
    return "│   " * depth


class DuanSSSP:
    def __init__(self, g, source, k=None, t=None, show_ties=False, preview=8, contar=False):
        self.g = g
        self.s = source
        n = g.n
        logn = math.log2(max(n, 2))
        # Seção 3.1: k = ⌊log^(1/3) n⌋, t = ⌊log^(2/3) n⌋
        self.k = k if k else max(1, math.floor(logn ** (1 / 3)))
        self.t = t if t else max(1, math.floor(logn ** (2 / 3)))
        # Nível do topo: l = ⌈log n / t⌉ (garantimos k·2^(lt) > n para que o
        # topo seja sempre uma execução bem-sucedida).
        self.top_level = max(1, math.ceil(logn / self.t))
        while self.k * 2 ** (self.top_level * self.t) <= n:
            self.top_level += 1

        self.d = [INF] * n          # d̂[v]
        self.h = [INF] * n          # nº de arestas do caminho (desempate, Assumption 2.1)
        self.pred = [-1] * n        # Pred[v]
        self.pred_w = [0] * n       # peso da aresta (Pred[v], v)
        self.d[source] = 0
        self.h[source] = 0

        self.show_ties = show_ties
        self.preview = preview
        self.info = log.isEnabledFor(logging.INFO)
        self.debug = log.isEnabledFor(logging.DEBUG)
        # contar=True liga as estimativas de comparações internas (heap, bisect, sort, min);
        # custa tempo, então o benchmark as colhe numa execução separada da cronometrada.
        self.contar = contar
        self.stats = dict(relaxacoes=0, re_relaxacoes_iguais=0, arestas_examinadas=0,
                          chamadas_bmssp=0, chamadas_base=0, chamadas_findpivots=0,
                          findpivots_desistencias=0, findpivots_S_total=0,
                          findpivots_W_total=0, findpivots_P_total=0,
                          pulls=0, inserts=0, prepends=0, heap_pushes=0, heap_pops=0,
                          comparacoes_cotas=0, adicoes=0, comparacoes_explicitas=0,
                          comparacoes_heap_est=0, comparacoes_D_est=0, comparacoes_total=0)
        self.niveis = {}  # l -> estatísticas das chamadas BMSSP naquele nível

    # ---- ordem total (Assumption 2.1) ------------------------------------- #
    def key(self, v):
        """Valor de d̂[v] na ordem total: (comprimento, nº de arestas, vértice)."""
        return (self.d[v], self.h[v], v)

    def _can_relax(self, u, v, w):
        """Condição "d̂[u] + w_uv <= d̂[v]" (Remark 3.4) na ordem total.
        O caminho candidato é ⟨d̂[u]+w, h[u]+1, v, u, ...⟩ e o atual é
        ⟨d̂[v], h[v], v, Pred[v], ...⟩; em empate decide-se por u vs Pred[v],
        e u == Pred[v] conta como igualdade (relaxação válida)."""
        return (self.d[u] + w, self.h[u] + 1, u) <= (self.d[v], self.h[v], self.pred[v])

    def _apply(self, u, v, w, depth):
        same = self.pred[v] == u and self.d[v] == self.d[u] + w and self.h[v] == self.h[u] + 1
        self.d[v] = self.d[u] + w
        self.h[v] = self.h[u] + 1
        self.pred[v] = u
        self.pred_w[v] = w
        self.stats["relaxacoes" if not same else "re_relaxacoes_iguais"] += 1
        if self.debug:
            log.debug("%s    relaxa %s -> %s (w=%g): d̂[%s] = %s%s", _indent(depth),
                      self.g.name(u), self.g.name(v), w, self.g.name(v), self.fmt(self.key(v)),
                      "  (igual: re-relaxação, Remark 3.4)" if same else "")

    # ---- formatação para os logs ------------------------------------------ #
    def fmt(self, key):
        if key[0] == INF:
            return "∞"
        if self.show_ties:
            return f"{key[0]:g}⟨h={key[1]},{self.g.name(key[2])}⟩"
        return f"{key[0]:g}"

    def fmt_set(self, S):
        names = sorted(S, key=self.key)
        shown = ", ".join(f"{self.g.name(v)}:{self.fmt(self.key(v))}" for v in names[:self.preview])
        extra = f", … (+{len(names) - self.preview})" if len(names) > self.preview else ""
        return "{" + shown + extra + "}"

    # ---- Algoritmo 1: FindPivots ------------------------------------------ #
    def find_pivots(self, B, S, depth):
        st = self.stats
        st["chamadas_findpivots"] += 1
        st["findpivots_S_total"] += len(S)
        ind = _indent(depth)
        k = self.k
        adj = self.g.adj
        if self.info:
            log.info("%s  FindPivots(B=%s, |S|=%d)", ind, self.fmt(B), len(S))
        W = set(S)
        prev = set(S)
        arestas = 0
        relax0 = st["relaxacoes"] + st["re_relaxacoes_iguais"]
        for i in range(1, k + 1):                       # relaxa por k passos
            Wi = set()
            for u in prev:
                adj_u = iter(adj[u])
                arestas += len(adj[u]) >> 1
                for v, w in zip(adj_u, adj_u):
                    if self._can_relax(u, v, w):
                        self._apply(u, v, w, depth)
                        if self.key(v) < B:
                            Wi.add(v)
            W |= Wi
            if self.debug:
                log.debug("%s    passo %d/%d: |W_%d| = %d, |W| = %d", ind, i, k, i, len(Wi), len(W))
            if len(W) > k * len(S):
                st["arestas_examinadas"] += arestas
                st["comparacoes_cotas"] += st["relaxacoes"] + st["re_relaxacoes_iguais"] - relax0  # d̂[v] < B
                st["findpivots_desistencias"] += 1
                st["findpivots_W_total"] += len(W)
                st["findpivots_P_total"] += len(S)
                if self.info:
                    log.info("%s    |W| = %d > k|S| = %d -> P = S (sem redução)", ind, len(W), k * len(S))
                return set(S), W
            prev = Wi
        st["arestas_examinadas"] += arestas
        st["comparacoes_cotas"] += st["relaxacoes"] + st["re_relaxacoes_iguais"] - relax0  # d̂[v] < B

        # F = {(u,v) : u,v ∈ W, d̂[v] = d̂[u] + w_uv}; sob a Assumption 2.1 é a
        # floresta dada por Pred[] restrita a W.
        children = {u: [] for u in W}
        has_parent = set()
        for v in W:
            u = self.pred[v]
            if u in children and self.d[v] == self.d[u] + self.pred_w[v] and self.h[v] == self.h[u] + 1:
                children[u].append(v)
                has_parent.add(v)

        P = set()
        for r in S:
            if r in has_parent:
                continue
            size, stack = 0, [r]
            while stack:
                x = stack.pop()
                size += 1
                stack.extend(children[x])
            if size >= k:
                P.add(r)
        st["findpivots_W_total"] += len(W)
        st["findpivots_P_total"] += len(P)
        st["comparacoes_cotas"] += len(W)             # teste de aresta justa, uma adição e uma comparação por v ∈ W
        st["adicoes"] += len(W)
        if self.info:
            log.info("%s    |W| = %d, pivôs P (árvores com >= k=%d vértices): %d de %d -> %s",
                     ind, len(W), k, len(P), len(S), self.fmt_set(P))
        return P, W

    # ---- Algoritmo 2: BaseCase -------------------------------------------- #
    def base_case(self, B, S, depth):
        st = self.stats
        st["chamadas_base"] += 1
        ind = _indent(depth)
        (x,) = S
        if self.info:
            log.info("%s  BaseCase(B=%s, S={%s}) -> mini-Dijkstra até k+1=%d vértices",
                     ind, self.fmt(B), self.g.name(x), self.k + 1)
        U0 = {x}
        heap = [(self.key(x), x)]
        in_heap = {x: self.key(x)}
        done = set()
        adj = self.g.adj
        contar = self.contar
        arestas = pushes = pops = comp_heap = cotas = 0
        while heap and len(U0) < self.k + 1:
            if contar:
                comp_heap += 2 * len(heap).bit_length()
            kv, u = heapq.heappop(heap)
            pops += 1
            if u in done or kv != self.key(u):
                continue  # entrada obsoleta (DecreaseKey preguiçoso)
            done.add(u)
            U0.add(u)
            if self.debug:
                log.debug("%s    extrai %s (d̂=%s)", ind, self.g.name(u), self.fmt(kv))
            adj_u = iter(adj[u])
            arestas += len(adj[u]) >> 1
            for v, w in zip(adj_u, adj_u):
                if self._can_relax(u, v, w):
                    cotas += 1                                       # d̂[u]+w < B
                    if (self.d[u] + w, self.h[u] + 1, v) < B:
                        self._apply(u, v, w, depth)
                        kv2 = self.key(v)
                        old = in_heap.get(v)
                        if old is not None:
                            cotas += 1                               # kv2 < in_heap[v]
                        if old is None or kv2 < old:
                            in_heap[v] = kv2
                            if contar:
                                comp_heap += (len(heap) + 1).bit_length()
                            heapq.heappush(heap, (kv2, v))
                            pushes += 1
        st["arestas_examinadas"] += arestas
        st["comparacoes_cotas"] += cotas
        st["heap_pushes"] += pushes
        st["heap_pops"] += pops
        st["comparacoes_heap_est"] += comp_heap
        if len(U0) <= self.k:
            if self.info:
                log.info("%s    -> sucesso: B' = B = %s, U = %s", ind, self.fmt(B), self.fmt_set(U0))
            return B, U0
        Bp = max(self.key(v) for v in U0)
        U = {v for v in U0 if self.key(v) < Bp}
        st["comparacoes_cotas"] += 2 * len(U0)
        if self.info:
            log.info("%s    -> parcial: B' = %s, U = %s", ind, self.fmt(Bp), self.fmt_set(U))
        return Bp, U

    # ---- Algoritmo 3: BMSSP ----------------------------------------------- #
    def _nivel(self, l):
        lv = self.niveis.get(l)
        if lv is None:
            lv = self.niveis[l] = dict(chamadas=0, parciais=0, iteracoes=0, S_total=0, S_max=0,
                                       U_total=0, U_max=0, S_cota=2 ** (l * self.t),
                                       U_cota=4 * self.k * 2 ** (l * self.t))
        return lv

    def bmssp(self, l, B, S, depth=0):
        st = self.stats
        st["chamadas_bmssp"] += 1
        lv = self._nivel(l)
        lv["chamadas"] += 1
        lv["S_total"] += len(S)
        if len(S) > lv["S_max"]:
            lv["S_max"] = len(S)
        ind = _indent(depth)
        if self.info:
            log.info("%s┌ BMSSP(l=%d, B=%s, |S|=%d) S=%s", ind, l, self.fmt(B), len(S), self.fmt_set(S))
        if l == 0:
            Bp, U = self.base_case(B, S, depth)
            lv["parciais"] += Bp < B
            lv["U_total"] += len(U)
            if len(U) > lv["U_max"]:
                lv["U_max"] = len(U)
            if self.info:
                log.info("%s└ fim BMSSP(l=0): B'=%s, |U|=%d", ind, self.fmt(Bp), len(U))
            return Bp, U

        P, W = self.find_pivots(B, S, depth)
        M = 2 ** ((l - 1) * self.t)
        D = BlockDS(M, B, self.fmt, depth, self.contar)
        for x in P:
            D.insert(x, self.key(x))
        Bp_last = min((self.key(x) for x in P), default=B)   # B'_0
        U = set()
        limit = self.k * 2 ** (l * self.t)
        adj = self.g.adj
        arestas = 0
        st["comparacoes_cotas"] += len(P)
        if self.info:
            log.info("%s  D inicializada com M=2^((l-1)t)=%d; limite |U| < k·2^(lt)=%d; B'_0=%s",
                     ind, M, limit, self.fmt(Bp_last))

        i = 0
        while len(U) < limit and len(D) > 0:
            i += 1
            Bi, Si = D.pull()
            st["pulls"] += 1
            if self.info:
                log.info("%s  [iter %d] Pull -> B_%d=%s, |S_%d|=%d (restam %d em D)",
                         ind, i, i, self.fmt(Bi), i, len(Si), len(D))
            Bpi, Ui = self.bmssp(l - 1, Bi, set(Si), depth + 1)
            U |= Ui

            K = []
            n_ins = 0
            relax0 = st["relaxacoes"] + st["re_relaxacoes_iguais"]
            for u in Ui:
                adj_u = iter(adj[u])
                arestas += len(adj[u]) >> 1
                for v, w in zip(adj_u, adj_u):
                    if self._can_relax(u, v, w):
                        self._apply(u, v, w, depth)
                        val = self.key(v)
                        if Bi <= val < B:
                            n_ins += D.insert(v, val)
                        elif Bpi <= val < Bi:
                            K.append((v, val))
            # por relaxação válida: "Bi <= val" e mais uma ("val < B" ou "Bpi <= val");
            # quem entra em K ainda testa "val < Bi"
            st["comparacoes_cotas"] += 2 * (st["relaxacoes"] + st["re_relaxacoes_iguais"] - relax0) + len(K)
            K.extend((x, self.key(x)) for x in Si if Bpi <= self.key(x) < Bi)
            st["comparacoes_cotas"] += 2 * len(Si)
            n_pre = D.batch_prepend(K)
            if self.info:
                log.info("%s  [iter %d] B'_%d=%s, |U_%d|=%d, |U|=%d; Insert em D: %d, BatchPrepend: %d",
                         ind, i, i, self.fmt(Bpi), i, len(Ui), len(U), n_ins, n_pre)
            Bp_last = Bpi

        Bp = min(Bp_last, B)
        extra = {x for x in W if self.key(x) < Bp} - U
        U |= extra
        st["arestas_examinadas"] += arestas
        st["comparacoes_cotas"] += len(W) + 1
        st["inserts"] += D.n_insert
        st["prepends"] += D.n_prepend
        st["comparacoes_D_est"] += D.comparacoes
        lv["iteracoes"] += i
        lv["parciais"] += Bp < B
        lv["U_total"] += len(U)
        if len(U) > lv["U_max"]:
            lv["U_max"] = len(U)
        if self.info:
            tipo = "sucesso" if Bp == B else "parcial (carga grande)"
            log.info("%s└ fim BMSSP(l=%d): %s, B'=%s, |U|=%d (+%d de W)",
                     ind, l, tipo, self.fmt(Bp), len(U), len(extra))
        return Bp, U

    def _fecha_contagem(self):
        """Operações do modelo comparação-adição (Seção 2), contadas exatamente.
        Explícitas: uma adição e uma comparação por aresta examinada (d̂[u]+w ≤ d̂[v]),
        uma comparação por pop do heap (teste de entrada obsoleta) e as comparações com
        as cotas B/Bi/B'i e o teste de aresta justa, acumuladas em comparacoes_cotas.
        Internas (só com contar=True): estimativas por operação, ⌈log₂ n⌉ por push e
        2⌈log₂ n⌉ por pop no heap, e bisect/sort/min na estrutura D."""
        st = self.stats
        st["adicoes"] += st["arestas_examinadas"]
        st["comparacoes_explicitas"] = st["arestas_examinadas"] + st["heap_pops"] + st["comparacoes_cotas"]
        st["comparacoes_total"] = (st["comparacoes_explicitas"] + st["comparacoes_heap_est"]
                                   + st["comparacoes_D_est"])

    def run(self):
        if self.info:
            log.info("Parâmetros: n=%d, m=%d, k=%d, t=%d, nível do topo l=%d",
                     self.g.n, self.g.m, self.k, self.t, self.top_level)
        self.bmssp(self.top_level, KEY_INF, {self.s})
        self._fecha_contagem()
        return self.d


# --------------------------------------------------------------------------- #
# Dijkstra (referência)
# --------------------------------------------------------------------------- #
def dijkstra(g, s, contar=False):
    """Dijkstra com heap binário e remoção preguiçosa. Com contar=True estima as
    comparações internas do heap (⌈log₂ n⌉ por push, 2⌈log₂ n⌉ por pop); o laço
    é duplicado para que a versão cronometrada não pague o teste."""
    d = [INF] * g.n
    d[s] = 0
    heap = [(0, s)]
    adj = g.adj
    relax = pops = arestas = comp_heap = 0
    pushes = 1
    if contar:
        while heap:
            comp_heap += 2 * len(heap).bit_length()
            du, u = heapq.heappop(heap)
            pops += 1
            if du > d[u]:
                continue
            adj_u = iter(adj[u])
            arestas += len(adj[u]) >> 1
            for v, w in zip(adj_u, adj_u):
                nd = du + w
                if nd < d[v]:
                    d[v] = nd
                    relax += 1
                    comp_heap += (len(heap) + 1).bit_length()
                    heapq.heappush(heap, (nd, v))
                    pushes += 1
    else:
        while heap:
            du, u = heapq.heappop(heap)
            pops += 1
            if du > d[u]:
                continue
            adj_u = iter(adj[u])
            arestas += len(adj[u]) >> 1
            for v, w in zip(adj_u, adj_u):
                nd = du + w
                if nd < d[v]:
                    d[v] = nd
                    relax += 1
                    heapq.heappush(heap, (nd, v))
                    pushes += 1
    stats = dict(relaxacoes=relax, pops=pops, pushes=pushes, arestas_examinadas=arestas,
                 adicoes=arestas, comparacoes_explicitas=arestas + pops,
                 comparacoes_heap_est=comp_heap)
    stats["comparacoes_total"] = stats["comparacoes_explicitas"] + comp_heap
    return d, stats


# --------------------------------------------------------------------------- #
# Grafos de teste
# --------------------------------------------------------------------------- #
def example_graph():
    names = list("sabcdefghij")
    g = Graph(len(names), names)
    idx = {c: i for i, c in enumerate(names)}
    for u, v, w in [("s", "a", 4), ("s", "b", 1), ("b", "a", 2), ("a", "c", 5),
                    ("b", "d", 8), ("c", "d", 1), ("d", "e", 3), ("c", "f", 6),
                    ("e", "f", 1), ("f", "g", 2), ("e", "h", 7), ("g", "h", 1),
                    ("h", "i", 2), ("g", "j", 9), ("i", "j", 1), ("j", "s", 3),
                    ("a", "b", 0), ("d", "c", 2)]:
        g.add_edge(idx[u], idx[v], w)
    return g, idx["s"]


def random_graph(n, m, seed, wmin, wmax, real):
    rng = random.Random(seed)
    g = Graph(n)
    order = list(range(1, n))
    rng.shuffle(order)
    seen = [0]
    draw = (lambda: rng.uniform(wmin, wmax)) if real else (lambda: rng.randint(wmin, wmax))
    for v in order:  # árvore aleatória com arcos nos dois sentidos: todo vértice alcança todos
        p, w = rng.choice(seen), draw()
        g.add_edge(p, v, w)
        g.add_edge(v, p, w)
        seen.append(v)
    while g.m < m:
        g.add_edge(rng.randrange(n), rng.randrange(n), draw())
    return g, 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="BMSSP (Duan et al. 2025) vs Dijkstra, com logs")
    ap.add_argument("--exemplo", action="store_true", help="usa o grafo pequeno fixo")
    ap.add_argument("--dimacs", metavar="ARQUIVO", help="carrega um grafo DIMACS .gr/.gr.gz (ignora -n, -m, --seed)")
    ap.add_argument("--fonte", type=int, default=1, help="vértice fonte do grafo DIMACS, 1-indexado como no arquivo")
    ap.add_argument("-n", type=int, default=50, help="vértices do grafo aleatório")
    ap.add_argument("-m", type=int, default=None, help="arestas (padrão: 3n)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--peso-min", type=float, default=1)
    ap.add_argument("--peso-max", type=float, default=20)
    ap.add_argument("--pesos-reais", action="store_true", help="pesos reais em vez de inteiros")
    ap.add_argument("--grau-constante", action="store_true",
                    help="aplica a transformação de grau constante da Seção 2")
    ap.add_argument("-k", type=int, default=None, help="força o parâmetro k")
    ap.add_argument("-t", type=int, default=None, help="força o parâmetro t")
    ap.add_argument("--log", default="INFO", help="TRACE, DEBUG, INFO, WARNING")
    ap.add_argument("--log-arquivo", default=None, help="grava o log também neste arquivo")
    ap.add_argument("--empates", action="store_true", help="mostra o desempate ⟨h,v⟩ nos valores")
    ap.add_argument("--preview", type=int, default=8, help="nº de vértices exibidos por conjunto")
    a = ap.parse_args()

    level = TRACE if a.log.upper() == "TRACE" else getattr(logging, a.log.upper())
    handlers = [logging.StreamHandler(sys.stdout)]
    if a.log_arquivo:
        handlers.append(logging.FileHandler(a.log_arquivo, mode="w", encoding="utf-8"))
    logging.basicConfig(level=level, format="%(message)s", handlers=handlers)

    if a.exemplo:
        g, s = example_graph()
    elif a.dimacs:
        t0 = time.perf_counter()
        g = load_gr(a.dimacs, Graph)
        if not 1 <= a.fonte <= g.n:
            ap.error(f"--fonte deve estar em [1, {g.n}]")
        s = a.fonte - 1
        log.info("DIMACS %s: n=%d, m=%d, carregado em %.1fs", a.dimacs, g.n, g.m, time.perf_counter() - t0)
    else:
        wmin, wmax = (a.peso_min, a.peso_max) if a.pesos_reais else (int(a.peso_min), int(a.peso_max))
        g, s = random_graph(a.n, a.m or 3 * a.n, a.seed, wmin, wmax, a.pesos_reais)

    work, ws, rep = g, s, list(range(g.n))
    if a.grau_constante:
        work, ws, rep = to_constant_degree(g, s)
        log.info("Transformação de grau constante: n %d -> %d, m %d -> %d", g.n, work.n, g.m, work.m)

    t0 = time.perf_counter()
    solver = DuanSSSP(work, ws, k=a.k, t=a.t, show_ties=a.empates, preview=a.preview)
    dw = solver.run()
    t_bmssp = time.perf_counter() - t0
    d_bmssp = [dw[rep[v]] for v in range(g.n)]

    t0 = time.perf_counter()
    d_dij, st_dij = dijkstra(g, s)
    t_dij = time.perf_counter() - t0

    bad = [v for v in range(g.n)
           if not (d_bmssp[v] == d_dij[v] or math.isclose(d_bmssp[v], d_dij[v], rel_tol=1e-9))]

    print("\n=== Resultado ===")
    if g.n <= 30:
        print(f"{'vértice':>8} {'BMSSP':>10} {'Dijkstra':>10}")
        for v in range(g.n):
            print(f"{g.name(v):>8} {d_bmssp[v]:>10g} {d_dij[v]:>10g}")
    print(f"BMSSP:    {t_bmssp * 1000:9.2f} ms  {solver.stats}")
    print(f"Dijkstra: {t_dij * 1000:9.2f} ms  {st_dij}")
    if bad:
        print(f"DIVERGÊNCIA em {len(bad)} vértice(s), ex.: "
              + ", ".join(f"{g.name(v)}: {d_bmssp[v]} vs {d_dij[v]}" for v in bad[:5]))
        sys.exit(1)
    print(f"OK: as {g.n} distâncias coincidem com Dijkstra.")


if __name__ == "__main__":
    main()
