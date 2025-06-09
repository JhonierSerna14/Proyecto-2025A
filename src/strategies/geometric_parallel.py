import numpy as np
import time
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from functools import partial

from src.models.base.sia import SIA
from src.controllers.manager import Manager
from src.funcs.format import fmt_biparticion
from src.middlewares.profile import profile, profiler_manager
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
    tot = marg.sum()
    return marg / tot if tot > 0 else marg


def run_partition_search(init_assign, n, dist_full, heuristic_per_var, dist_metric, S_data):
    marg_cache = {}

    def get_marg(idx_tuple: Tuple[int, ...]) -> np.ndarray:
        if idx_tuple not in marg_cache:
            marg_cache[idx_tuple] = marginalizar_subconjunto(dist_full, list(idx_tuple))
        return marg_cache[idx_tuple]

    local_best_score = float("inf")
    local_best_part = None
    local_best_distP = None

    def _search_part(idx: int, mech: List[int], alc: List[int]):
        nonlocal local_best_score, local_best_part, local_best_distP

        if mech and get_marg(tuple(mech)).sum() == 0:
            return
        if alc and get_marg(tuple(alc)).sum() == 0:
            return

        rem = n - idx
        bound = rem * heuristic_per_var
        if bound >= local_best_score:
            return

        if idx == n:
            if not mech or not alc:
                return
            part = S_data.bipartir(np.array(alc, dtype=int), np.array(mech, dtype=int))
            distP = part.distribucion_marginal()
            mP_mech = marginalizar_subconjunto(distP, mech)
            mP_alc = marginalizar_subconjunto(distP, alc)
            score = dist_metric(mP_mech, get_marg(tuple(mech))) + \
                    dist_metric(mP_alc, get_marg(tuple(alc)))
            if score < local_best_score:
                local_best_score = score
                local_best_part = (mech.copy(), alc.copy())
                local_best_distP = distP
            return

        mech.append(idx)
        _search_part(idx + 1, mech, alc)
        mech.pop()

        alc.append(idx)
        _search_part(idx + 1, mech, alc)
        alc.pop()

    mech, alc = [], []
    for i, val in enumerate(init_assign):
        (mech if val == 1 else alc).append(i)
    _search_part(len(init_assign), mech, alc)
    return (local_best_score, local_best_part, local_best_distP)


class GeometricParallelSIA(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(f"GP{len(gestor.estado_inicial)}{gestor.pagina}")
        self.dist = seleccionar_metrica(aplicacion.distancia_metrica)

    @profile(name="GeometricP")
    def aplicar_estrategia(self, condicion: str, alcance: str, mecanismo: str) -> Solution:
        t0 = time.time()
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        S = self.sia_subsistema
        n = len(S.indices_ncubos)

        dist_full = S.distribucion_marginal()
        assert dist_full.sum() > 0, "¡Distribución del subsistema es cero!"

        diffs = np.abs(dist_full - np.roll(dist_full, 1))
        heuristic_per_var = np.percentile(diffs, 5)

        k = 3  # nivel de profundidad para ramificar
        initial_assignments = list(product([0, 1], repeat=k))  # 0=alc, 1=mech

        search_fn = partial(
            run_partition_search,
            n=n,
            dist_full=dist_full,
            heuristic_per_var=heuristic_per_var,
            dist_metric=self.dist,
            S_data=S
        )

        with ProcessPoolExecutor() as executor:
            results = list(executor.map(search_fn, initial_assignments))

        best_score, best_part, best_distP = min(
            (r for r in results if r[1] is not None),
            key=lambda x: x[0],
            default=(float('inf'), None, None)
        )

        if best_part is None:
            raise ValueError("No se encontró bipartición válida.")

        mech, alc = best_part
        idx_pres = set(S.dims_ncubos.data)
        idx_futu = set(S.indices_ncubos.data)
        dual_mech = list(idx_pres - set(mech))
        dual_alc = list(idx_futu - set(alc))
        fmt = fmt_biparticion([mech, alc], [dual_mech, dual_alc])

        return Solution(
            estrategia="GeometricSIA-Parallel",
            perdida=best_score,
            distribucion_subsistema=dist_full,
            distribucion_particion=best_distP,
            tiempo_total=time.time() - t0,
            particion=fmt
        )
