import numpy as np
import time
from typing import List, Tuple
from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.funcs.format import fmt_biparticion
from src.models.core.solution import Solution
from src.funcs.base import seleccionar_metrica
from src.models.base.application import aplicacion

def marginalizar_subconjunto(full_dist: np.ndarray, indices: List[int]) -> np.ndarray:
    n = int(np.log2(full_dist.size))
    k = len(indices)
    if k == n:
        v = full_dist.copy()
        return v/ v.sum() if v.sum()>0 else v
    marg = np.zeros(1<<k, dtype=float)
    for m in range(1<<n):
        idx = 0
        for pos, var in enumerate(indices):
            bit = (m >> var) & 1
            idx |= (bit << pos)
        marg[idx] += full_dist[m]
    tot = marg.sum()
    return marg/tot if tot>0 else marg

class GeometricSIA(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        self.dist = seleccionar_metrica(aplicacion.distancia_metrica)

    def aplicar_estrategia(self, condicion: str, alcance: str, mecanismo: str) -> Solution:
        t0 = time.time()
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        S = self.sia_subsistema
        n = len(S.indices_ncubos)

        # Distribución completa
        dist_full = S.distribucion_marginal()
        assert dist_full.sum()>0, "¡Distribución del subsistema es cero!"

        # Cache de marginales
        marg_cache = {}
        def get_marg(idx_tuple: Tuple[int,...]) -> np.ndarray:
            if idx_tuple not in marg_cache:
                marg_cache[idx_tuple] = marginalizar_subconjunto(dist_full, list(idx_tuple))
            return marg_cache[idx_tuple]

        # Cota heurística: mínimo coste de transicionar por variable
        # (usamos percentil 5% de las diferencias absolutas como aprox.)
        diffs = np.abs(dist_full - np.roll(dist_full,1))
        heuristic_per_var = np.percentile(diffs, 5)

        best_score = float('inf')
        best_part  = None
        best_distP = None

        def _search_part(idx: int, mech: List[int], alc: List[int]):
            nonlocal best_score, best_part, best_distP

            # Poda trivial: si mech o alc ya no tienen masa, no sigue
            if mech:
                if get_marg(tuple(mech)).sum() == 0:
                    return
            if alc:
                if get_marg(tuple(alc)).sum() == 0:
                    return

            # Cota parcial: EMD parcial es 0, más heurística para las variables restantes
            rem = n - idx
            bound = rem * heuristic_per_var
            if bound >= best_score:
                return

            # Si ya asignamos todas las variables, evaluamos
            if idx == n:
                # descartamos particiones triviales
                if not mech or not alc:
                    return
                # calcula EMD final
                part = S.bipartir(np.array(alc,dtype=int), np.array(mech,dtype=int))
                distP = part.distribucion_marginal()
                mP_mech = marginalizar_subconjunto(distP, mech)
                mP_alc  = marginalizar_subconjunto(distP, alc)
                score = self.dist(mP_mech, get_marg(tuple(mech))) + \
                        self.dist(mP_alc,  get_marg(tuple(alc)))
                if score < best_score:
                    best_score = score
                    best_part  = (mech.copy(), alc.copy())
                    best_distP = distP
                return

            # Asignar variable idx a mecanismo
            mech.append(idx)
            _search_part(idx+1, mech, alc)
            mech.pop()

            # Asignar variable idx a alcance
            alc.append(idx)
            _search_part(idx+1, mech, alc)
            alc.pop()

        # Inicio de la búsqueda
        _search_part(0, [], [])

        if best_part is None:
            raise ValueError("No se encontró bipartición válida.")

        mech, alc = best_part
        idx_pres = set(S.dims_ncubos.data)
        idx_futu = set(S.indices_ncubos.data)
        dual_mech = list(idx_pres - set(mech))
        dual_alc  = list(idx_futu - set(alc))
        fmt = fmt_biparticion([mech, alc], [dual_mech, dual_alc])

        return Solution(
            estrategia="GeometricSIA-BnB+Poda",
            perdida=best_score,
            distribucion_subsistema=dist_full,
            distribucion_particion=best_distP,
            tiempo_total=time.time()-t0,
            particion=fmt
        )
