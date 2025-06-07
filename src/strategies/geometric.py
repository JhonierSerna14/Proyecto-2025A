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
    """
    full_dist: vector de tamaño 2^n con la dist. conjunta sobre {0,1}^n.
    indices: lista de posiciones (0…n-1) de las variables que queremos conservar.
    Devuelve la marginal normalizada de tamaño 2^k.
    """
    n = int(np.log2(full_dist.size))
    k = len(indices)
    # caso trivial: el sistema ya es el subsistema
    if k == n:
        v = full_dist.copy()
        return v / v.sum() if v.sum() > 0 else v

    size_k = 1 << k
    marg = np.zeros(size_k, dtype=float)

    # para cada configuración m de las n variables
    for m in range(1 << n):
        # construimos el índice de la marginal extrayendo bit a bit
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
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        S = self.sia_subsistema
        n = len(S.indices_ncubos)

        # precalcula la distribución completa y sus marginales unitarias
        dist_full = S.distribucion_marginal()
        # cache de marginales de cualquier subconjunto para poda rápida
        marg_cache = {}

        def get_marg(indices: Tuple[int,...]) -> np.ndarray:
            if indices not in marg_cache:
                marg_cache[indices] = marginalizar_subconjunto(dist_full, list(indices))
            return marg_cache[indices]

        best_score = float('inf')
        best_part  = None
        best_distP = None

        # función recursiva de búsqueda con poda
        def _search_part(idx: int,
                         mech: List[int],
                         alc:  List[int]):
            nonlocal best_score, best_part, best_distP

            # Bound parcial: calcula EMD sólo para la parte completamente asignada
            # y usa 0 como cota para no asignados
            if mech:
                mS_mech = get_marg(tuple(mech))
                # como mP_mech aún no existe, la mejor correspondencia es mS_mech misma (dist=0)
                bound_mech = 0.0
            else:
                bound_mech = 0.0
            if alc:
                mS_alc = get_marg(tuple(alc))
                bound_alc = 0.0
            else:
                bound_alc = 0.0
            # cota global
            if bound_mech + bound_alc >= best_score:
                return  # poda inmediata

            # si ya asigné todas las variables
            if idx == n:
                # evalúa completamente esta partición
                mP_mech = marginalizar_subconjunto(S.bipartir(
                    np.array(mech,dtype=int), np.array(alc,dtype=int)
                ).distribucion_marginal(), mech)
                mP_alc  = marginalizar_subconjunto(S.bipartir(
                    np.array(mech,dtype=int), np.array(alc,dtype=int)
                ).distribucion_marginal(), alc)
                score = self.dist(mP_mech, get_marg(tuple(mech))) + \
                        self.dist(mP_alc,  get_marg(tuple(alc)))
                if score < best_score:
                    best_score = score
                    best_part  = (mech.copy(), alc.copy())
                    best_distP = S.bipartir(
                        np.array(mech,dtype=int), np.array(alc,dtype=int)
                    ).distribucion_marginal()
                return

            # asignar variable idx a mecanismo (0)
            mech.append(idx)
            _search_part(idx+1, mech, alc)
            mech.pop()

            # asignar variable idx a alcance (1)
            alc.append(idx)
            _search_part(idx+1, mech, alc)
            alc.pop()

        # iniciar búsqueda
        _search_part(0, [], [])

        if best_part is None:
            raise ValueError("No se encontró bipartición válida.")

        mech, alc = best_part
        # duales
        idx_pres = set(S.dims_ncubos.data)
        idx_futu = set(S.indices_ncubos.data)
        dual_mech = list(idx_pres - set(mech))
        dual_alc  = list(idx_futu - set(alc))
        fmt = fmt_biparticion([mech, alc], [dual_mech, dual_alc])

        return Solution(
            estrategia="GeometricSIA-BnB",
            perdida=best_score,
            distribucion_subsistema=dist_full,
            distribucion_particion=best_distP,
            tiempo_total=time.time()-t0,
            particion=fmt
        )
