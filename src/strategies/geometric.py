import numpy as np
import time
from collections import deque
from itertools import combinations, permutations
from typing import List, Tuple, Dict, Set

from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.funcs.format import fmt_biparticion
from src.models.core.solution import Solution
from src.funcs.base import seleccionar_metrica
from src.models.base.application import aplicacion


def marginalizar_subconjunto(full_dist: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    Marginaliza full_dist (de tamaño 2^n) sobre las variables en 'indices'.
    Devuelve vector de tamaño 2^k normalizado.
    """
    n = int(np.log2(full_dist.size))
    k = len(indices)
    if k == n:
        v = full_dist.copy()
        return v / v.sum() if v.sum() > 0 else v
    marg = np.zeros(1 << k, dtype=float)
    for m in range(1 << n):
        bits = ((m >> np.array(indices)) & 1).astype(int)
        idx = sum(bit << pos for pos, bit in enumerate(bits))
        marg[idx] += full_dist[m]
    total = marg.sum()
    return marg / total if total > 0 else marg


class GeometricSIA(SIA):
    """
    Estrategia geométrica optimizada según el artículo:
      1) BFS nivel-a-nivel para cada par (i,j), sólo expandiendo vecinos que
         reduzcan la distancia de Hamming a j.
      2) Explotación completa de simetrías (permutaciones de ejes y complemento de bits).
      3) Heurística set‐cover en zonas de bajo coste.
    """

    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        self.distancia = seleccionar_metrica(aplicacion.distancia_metrica)

    def calcular_transicion_coste(
        self, i: int, j: int, tensor_v: np.ndarray, n: int
    ) -> float:
        """
        Calcula t_v(i,j) con BFS nivel-a-nivel, expandiendo solo vecinos
        que disminuyan la distancia de Hamming a j, y memoizando costos
        bajo simetrías (complemento de bits y permutaciones de ejes).
        """
        # Generar todas las simetrías del hipercubo
        coords = list(range(n))

        def canonical(a: int, b: int) -> Tuple[int, int]:
            reps = []
            for perm in permutations(coords):
                for comp in (False, True):

                    def transform(x: int) -> int:
                        bits = [(x >> k) & 1 for k in range(n)]
                        permuted = [bits[k] for k in perm]
                        if comp:
                            permuted = [1 - bit for bit in permuted]
                        return sum(bit << idx for idx, bit in enumerate(permuted))

                    ai, bi = transform(a), transform(b)
                    reps.append(tuple(sorted((ai, bi))))
            return min(reps)

        target = j
        memo: Dict[Tuple[int, int], float] = {}
        # BFS queue of (state, cost_so_far)
        queue = deque([(i, 0.0)])
        best = float("inf")
        while queue:
            state, cost = queue.popleft()
            if state == target:
                best = min(best, cost)
                continue
            d_curr = bin(state ^ target).count("1")
            for bit in range(n):
                neigh = state ^ (1 << bit)
                # sólo vecinos que acerquen a j
                if bin(neigh ^ target).count("1") == d_curr - 1:
                    gamma = 2.0 ** (-d_curr)
                    c_edge = gamma * abs(tensor_v[state] - tensor_v[neigh])
                    new_cost = cost + c_edge
                    key = canonical(state, neigh)
                    if key not in memo or new_cost < memo[key]:
                        memo[key] = new_cost
                        queue.append((neigh, new_cost))
        return best

    def _construir_tabla_costes(
        self, tensores: Dict[int, np.ndarray], n: int
    ) -> Dict[int, Dict[Tuple[int, int], float]]:
        """
        Para cada variable v, construye T[v][(i,j)] = t_v(i,j) invocando
        calcular_transicion_coste para cada par (i<j), y rellena simétricamente.
        """
        size = 1 << n
        T: Dict[int, Dict[Tuple[int, int], float]] = {}
        for v, tensor in tensores.items():
            Tv: Dict[Tuple[int, int], float] = {}
            for i in range(size):
                for j in range(i, size):
                    if i == j:
                        cost = 0.0
                    else:
                        cost = self.calcular_transicion_coste(i, j, tensor, n)
                    Tv[(i, j)] = cost
                    Tv[(j, i)] = cost
            T[v] = Tv
        return T

    def identificar_candidatos(
        self, T: Dict[int, Dict[Tuple[int, int], float]], n: int
    ) -> List[Tuple[List[int], List[int]]]:
        # sin cambios respecto a tu versión
        zonas: Dict[int, Set[int]] = {}
        for v, costes in T.items():
            arr = np.array(list(costes.values()))
            thr = np.percentile(arr, 5)
            zonas[v] = {j for (i, j), c in costes.items() if c <= thr}
        candidatos: List[Tuple[List[int], List[int]]] = []
        vars_list = list(zonas)
        k_max = min(4, len(vars_list))
        for k in range(1, k_max + 1):
            for combo in combinations(vars_list, k):
                inter = set.intersection(*(zonas[v] for v in combo))
                for j in list(inter)[:5]:
                    bits = [(j >> b) & 1 for b in range(n)]
                    mech = [idx for idx, bit in enumerate(bits) if bit == 0]
                    alc = [idx for idx, bit in enumerate(bits) if bit == 1]
                    if mech and alc:
                        candidatos.append((mech, alc))
        seen = set()
        uniq = []
        for mech, alc in candidatos:
            key = (tuple(mech), tuple(alc))
            if key not in seen:
                seen.add(key)
                uniq.append((mech, alc))
        return uniq

    def aplicar_estrategia(
        self, condicion: str, alcance: str, mecanismo: str
    ) -> Solution:
        # sin cambios en la invocación, mantiene tu firma
        t0 = time.time()
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        S = self.sia_subsistema
        n = len(S.indices_ncubos)
        tensores = {nc.indice: nc.data.flatten() for nc in S.ncubos}
        T = self._construir_tabla_costes(tensores, n)
        candidatos = self.identificar_candidatos(T, n)
        dist_full = S.distribucion_marginal()
        best_score, best_part, best_dist = float("inf"), None, None
        for mech, alc in candidatos:
            part = S.bipartir(np.array(alc), np.array(mech))
            mS_mech = marginalizar_subconjunto(dist_full, mech)
            mS_alc = marginalizar_subconjunto(dist_full, alc)
            distP = part.distribucion_marginal()
            mP_mech = marginalizar_subconjunto(distP, mech)
            mP_alc = marginalizar_subconjunto(distP, alc)
            score = self.distancia(mP_mech, mS_mech) + self.distancia(mP_alc, mS_alc)
            if score < best_score:
                best_score, best_part, best_dist = score, (mech, alc), distP
        if best_part is None:
            raise ValueError("No se encontró bipartición óptima.")
        mech, alc = best_part
        # Calcular duales según subsistema
        idx_pres = set(S.dims_ncubos.data)
        idx_futu = set(S.indices_ncubos.data)
        dual_mech = list(idx_pres - set(mech))
        dual_alc = list(idx_futu - set(alc))
        fmt = fmt_biparticion([mech, alc], [dual_mech, dual_alc])
        return Solution(
            estrategia="GeometricSIA",
            perdida=best_score,
            distribucion_subsistema=dist_full,
            distribucion_particion=best_dist,
            tiempo_total=time.time() - t0,
            particion=fmt,
        )
