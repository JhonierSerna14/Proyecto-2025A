import numpy as np
import time
from heapq import nsmallest, heappush
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

class GeometricSIA(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        self.dist = seleccionar_metrica(aplicacion.distancia_metrica)

    def aplicar_estrategia(self, condicion: str, alcance: str, mecanismo: str) -> Solution:
        t0 = time.time()
        # 1) Prepara subsistema
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        S = self.sia_subsistema
        n = len(S.indices_ncubos)

        # 2) Distribución completa
        dist_full = S.distribucion_marginal()
        assert dist_full.sum() > 0, "¡Distribución del subsistema es cero!"

        # 3) Cache de marginales
        marg_cache = {}
        def get_marg(idx_tuple: Tuple[int, ...]) -> np.ndarray:
            if idx_tuple not in marg_cache:
                marg_cache[idx_tuple] = marginalizar_subconjunto(dist_full, list(idx_tuple))
            return marg_cache[idx_tuple]

        # 4) Heurística por variable restante
        diffs = np.abs(dist_full - np.roll(dist_full, 1))
        heuristic_per_var = np.percentile(diffs[diffs > 0], 5) if np.any(diffs > 0) else 0.0

        # 5) Greedy init (no trivial) y partición de respaldo
        def greedy_init() -> Tuple[float, Tuple[int,...], Tuple[int,...], np.ndarray]:
            mech, alc = [], []
            for i in range(n):
                m_i = marginalizar_subconjunto(dist_full, [i])
                p1 = m_i[-1] if m_i.size > 1 else m_i[0]
                if p1 > 0.5:
                    mech.append(i)
                else:
                    alc.append(i)
            # fuerza no trivial
            if not mech:
                mech.append(alc.pop())
            elif not alc:
                alc.append(mech.pop())
            # evalúa phi y distP
            part = S.bipartir(np.array(alc, dtype=int), np.array(mech, dtype=int))
            distP = part.distribucion_marginal()
            mP_mech = marginalizar_subconjunto(distP, mech)
            mP_alc  = marginalizar_subconjunto(distP, alc)
            phi = self.dist(mP_mech, get_marg(tuple(mech))) + \
                  self.dist(mP_alc,  get_marg(tuple(alc)))
            return phi, tuple(mech), tuple(alc), distP

        best_score, best_mech, best_alc, best_distP = greedy_init()

        # 6) Variable ordering por varianza de marginal univariable
        impacts = []
        for i in range(n):
            marg_i = marginalizar_subconjunto(dist_full, [i])
            impacts.append((i, float(np.var(marg_i))))
        order = [i for i,_ in sorted(impacts, key=lambda x: -x[1])]

        # 7) Beam Search + Branch & Bound
        B = 200
        beam: List[Tuple[float,int,Tuple[int,...],Tuple[int,...]]] = [(0.0, 0, (), ())]

        while True:
            if all(pos == n for _, pos, _, _ in beam):
                break
            next_beam = []
            for bound, pos, mech_tup, alc_tup in beam:
                if pos < n:
                    # poda trivial
                    if mech_tup and get_marg(mech_tup).sum() == 0: continue
                    if alc_tup and get_marg(alc_tup).sum() == 0:   continue
                    # cota parcial
                    rem = n - pos
                    bp = rem * heuristic_per_var
                    if bp >= best_score: continue
                    var = order[pos]
                    heappush(next_beam, (bp, pos+1, mech_tup + (var,), alc_tup))
                    heappush(next_beam, (bp, pos+1, mech_tup, alc_tup + (var,)))
                else:
                    heappush(next_beam, (bound, pos, mech_tup, alc_tup))
            beam = nsmallest(B, next_beam, key=lambda x: x[0])

        # 8) Evaluar estados finales
        for _, pos, mech_tup, alc_tup in beam:
            if not mech_tup or not alc_tup: 
                continue
            part = S.bipartir(np.array(list(alc_tup), dtype=int),
                              np.array(list(mech_tup), dtype=int))
            distP = part.distribucion_marginal()
            mP_mech = marginalizar_subconjunto(distP, list(mech_tup))
            mP_alc  = marginalizar_subconjunto(distP, list(alc_tup))
            phi = self.dist(mP_mech, get_marg(mech_tup)) + \
                  self.dist(mP_alc,  get_marg(alc_tup))
            if phi < best_score:
                best_score, best_mech, best_alc, best_distP = (
                    phi, mech_tup, alc_tup, distP
                )

        # 9) Formatear partición y devolver
        idx_pres = set(S.dims_ncubos.data)
        idx_futu = set(S.indices_ncubos.data)
        dual_mech = list(idx_pres - set(best_mech))
        dual_alc  = list(idx_futu - set(best_alc))
        fmt = fmt_biparticion([list(best_mech), list(best_alc)], [dual_mech, dual_alc])

        return Solution(
            estrategia="GeometricSIA",
            perdida=best_score,
            distribucion_subsistema=dist_full,
            distribucion_particion=best_distP,
            tiempo_total=time.time() - t0,
            particion=fmt
        )
