"""
Módulo GeometricSIA - Implementación de estrategia geométrica para búsqueda de biparticiones óptimas

Este módulo implementa una estrategia de búsqueda de biparticiones óptimas para sistemas de información integrada (SIA)
utilizando un enfoque geométrico y simétrico. La implementación hereda de la clase base SIA y proporciona una solución
sofisticada para encontrar particiones que minimicen la pérdida de información.

Características principales:
- Búsqueda por haz (beam search) con poda heurística
- Manejo de simetrías del sistema
- Optimización de memoria mediante caché
- Múltiples inicializaciones greedy
- Métricas de calidad combinadas

Autor: [Autor del proyecto]
Fecha: [Fecha de implementación]
"""

import numpy as np
import time
from heapq import nsmallest, heappush
from typing import List, Tuple, Set
from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.funcs.format import fmt_biparticion
from src.models.core.solution import Solution
from src.funcs.base import seleccionar_metrica
from src.models.base.application import aplicacion


def marginalizar_subconjunto(full_dist: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    Calcula la distribución marginal para un subconjunto de variables.

    Args:
        full_dist (np.ndarray): Distribución completa del sistema
        indices (List[int]): Lista de índices de variables a marginalizar

    Returns:
        np.ndarray: Distribución marginal normalizada para el subconjunto especificado
    """
    n = int(np.log2(full_dist.size))
    k = len(indices)
    if k == n:
        v = full_dist.copy()
        return v / v.sum() if v.sum() > 0 else v
    marg = np.zeros(1 << k, dtype=float)
    for m in range(1 << n):
        idx = 0
        for pos, var in enumerate(indices):
            bit = (m >> var) & 1
            idx |= (bit << pos)
        marg[idx] += full_dist[m]
    total = marg.sum()
    return marg / total if total > 0 else marg


def canonical_form(mech: Tuple[int, ...], alc: Tuple[int, ...], n: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Encuentra la forma canónica de una partición bajo permutaciones de coordenadas.
    Explora el grupo generado por transposiciones adyacentes y el intercambio mech<->alc
    para encontrar la partición lexicográficamente más pequeña.

    Args:
        mech (Tuple[int, ...]): Tupla de índices del mecanismo
        alc (Tuple[int, ...]): Tupla de índices del alcance
        n (int): Número total de variables

    Returns:
        Tuple[Tuple[int, ...], Tuple[int, ...]]: Forma canónica de la partición
    """
    mech0 = tuple(sorted(mech))
    alc0 = tuple(sorted(alc))
    initial = (mech0, alc0)
    best = initial
    visited = {initial}
    queue = [initial]

    while queue:
        mech_c, alc_c = queue.pop(0)
        # update best lexicographically
        if mech_c < best[0] or (mech_c == best[0] and alc_c < best[1]):
            best = (mech_c, alc_c)
        # neighbor: swap mech and alc
        swap = (tuple(sorted(alc_c)), tuple(sorted(mech_c)))
        if swap not in visited:
            visited.add(swap)
            queue.append(swap)
        # neighbors: adjacent coordinate swaps
        for i in range(n - 1):
            perm = list(range(n))
            perm[i], perm[i+1] = perm[i+1], perm[i]
            mech_p = tuple(sorted(perm[j] for j in mech_c))
            alc_p = tuple(sorted(perm[j] for j in alc_c))
            nb = (mech_p, alc_p)
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return best


def hamming_distance_partition(mech: Tuple[int, ...], alc: Tuple[int, ...], n: int) -> int:
    """
    Calcula la distancia de Hamming entre particiones.

    Args:
        mech (Tuple[int, ...]): Tupla de índices del mecanismo
        alc (Tuple[int, ...]): Tupla de índices del alcance
        n (int): Número total de variables

    Returns:
        int: Distancia de Hamming entre las particiones
    """
    vec = np.zeros(n, dtype=int)
    vec[list(mech)] = 1
    return int(np.sum(vec != (1 - vec)))


class GeometricSIA(SIA):
    """
    Implementación de estrategia geométrica para búsqueda de biparticiones óptimas.
    
    Esta clase implementa un algoritmo de búsqueda por haz (beam search) con poda heurística
    y manejo de simetrías para encontrar biparticiones óptimas en sistemas de información integrada.

    Args:
        gestor (Manager): Gestor del sistema
        beam_size (int, optional): Tamaño del haz de búsqueda. Defaults to 200.
        heuristic_percentile (float, optional): Percentil para la heurística. Defaults to 5.0.
        alpha (float, optional): Factor de penalización para la distancia de Hamming. Defaults to 0.1.
    """

    def __init__(self, gestor: Manager,
                 beam_size: int = 200,
                 heuristic_percentile: float = 5.0,
                 alpha: float = 0.1):
        super().__init__(gestor)
        self.dist = seleccionar_metrica(aplicacion.distancia_metrica)
        self.B = beam_size
        self.pctl = heuristic_percentile
        self.alpha = alpha

    def greedy_init_multiple(self, dist_full: np.ndarray, n: int) -> List[Tuple[float, Tuple[int,...], Tuple[int,...], np.ndarray]]:
        """
        Genera múltiples soluciones iniciales usando diferentes umbrales.

        Args:
            dist_full (np.ndarray): Distribución completa del sistema
            n (int): Número total de variables

        Returns:
            List[Tuple[float, Tuple[int,...], Tuple[int,...], np.ndarray]]: Lista de soluciones iniciales
            con sus respectivas métricas de calidad
        """
        thresholds = [0.5, 0.6, 0.4]
        results = []
        
        for threshold in thresholds:
            mech, alc = [], []
            for i in range(n):
                m_i = marginalizar_subconjunto(dist_full, [i])
                p1 = m_i[-1] if m_i.size > 1 else m_i[0]
                if p1 > threshold:
                    mech.append(i)
                else:
                    alc.append(i)
            
            if not mech:
                mech.append(alc.pop())
            elif not alc:
                alc.append(mech.pop())

            part = self.sia_subsistema.bipartir(np.array(alc, dtype=int), np.array(mech, dtype=int))
            distP = part.distribucion_marginal()
            mP_mech = marginalizar_subconjunto(distP, mech)
            mP_alc  = marginalizar_subconjunto(distP, alc)
            hdist = hamming_distance_partition(tuple(mech), tuple(alc), n)
            phi = (self.dist(mP_mech, marginalizar_subconjunto(dist_full, mech))
                   + self.dist(mP_alc, marginalizar_subconjunto(dist_full, alc))) * np.exp(-self.alpha * hdist)
            results.append((phi, tuple(mech), tuple(alc), distP))
        
        return results

    def aplicar_estrategia(self, condicion: str, alcance: str, mecanismo: str) -> Solution:
        """
        Implementa el algoritmo principal de búsqueda de biparticiones óptimas.

        Args:
            condicion (str): Condición de fondo del sistema
            alcance (str): Alcance del sistema
            mecanismo (str): Mecanismo del sistema

        Returns:
            Solution: Solución encontrada con la mejor bipartición
        """
        t0 = time.time()
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        S = self.sia_subsistema
        n = len(S.indices_ncubos)

        dist_full = S.distribucion_marginal()
        assert dist_full.sum() > 0, "¡Distribución del subsistema es cero!"

        marg_cache = {}
        def get_marg(idx_tuple: Tuple[int, ...]) -> np.ndarray:
            if idx_tuple not in marg_cache:
                marg_cache[idx_tuple] = marginalizar_subconjunto(dist_full, list(idx_tuple))
            return marg_cache[idx_tuple]

        diffs = np.abs(dist_full - np.roll(dist_full, 1))
        heuristic_per_var = np.percentile(diffs[diffs > 0], self.pctl) if np.any(diffs > 0) else 0.0

        init_results = self.greedy_init_multiple(dist_full, n)
        best_score, best_mech, best_alc, best_distP = min(init_results, key=lambda x: x[0])

        impacts = [(i, float(np.var(marginalizar_subconjunto(dist_full, [i])))) for i in range(n)]
        order = [i for i,_ in sorted(impacts, key=lambda x: -x[1])]

        beam = [(0.0, 0, (), ())]
        visited = set()
        
        while True:
            if all(pos == n for _, pos, _, _ in beam):
                break
            next_beam = []
            for bound, pos, mech_tup, alc_tup in beam:
                if pos < n:
                    if mech_tup and get_marg(mech_tup).sum() == 0: continue
                    if alc_tup and get_marg(alc_tup).sum() == 0: continue
                    rem = n - pos
                    bp = rem * heuristic_per_var
                    if bp >= best_score: continue
                    var = order[pos]
                    for mech_new, alc_new in [(mech_tup + (var,), alc_tup), (mech_tup, alc_tup + (var,))]:
                        key = canonical_form(mech_new, alc_new, n)
                        if key in visited: continue
                        visited.add(key)
                        heappush(next_beam, (bp, pos+1, mech_new, alc_new))
                else:
                    heappush(next_beam, (bound, pos, mech_tup, alc_tup))
            beam = nsmallest(self.B, next_beam, key=lambda x: x[0])

        for _, _, mech_tup, alc_tup in beam:
            if not mech_tup or not alc_tup:
                continue
            part = S.bipartir(np.array(list(alc_tup), dtype=int),
                              np.array(list(mech_tup), dtype=int))
            distP = part.distribucion_marginal()
            mP_mech = marginalizar_subconjunto(distP, list(mech_tup))
            mP_alc  = marginalizar_subconjunto(distP, list(alc_tup))
            hdist = hamming_distance_partition(mech_tup, alc_tup, n)
            phi = (self.dist(mP_mech, get_marg(mech_tup))
                   + self.dist(mP_alc, get_marg(alc_tup))) * np.exp(-self.alpha * hdist)
            if phi < best_score:
                best_score, best_mech, best_alc, best_distP = phi, mech_tup, alc_tup, distP

        idx_pres = set(S.dims_ncubos.data)
        idx_futu = set(S.indices_ncubos)
        dual_mech = list(idx_pres - set(best_mech))
        dual_alc = list(idx_futu - set(best_alc))
        fmt = fmt_biparticion([list(best_mech), list(best_alc)], [dual_mech, dual_alc])

        return Solution(
            estrategia="Geometric",
            perdida=best_score,
            distribucion_subsistema=dist_full,
            distribucion_particion=best_distP,
            tiempo_total=time.time() - t0,
            particion=fmt
        )
