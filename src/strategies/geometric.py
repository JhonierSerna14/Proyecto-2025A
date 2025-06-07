import numpy as np
import time
from collections import deque
from itertools import combinations, permutations, product
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
    bit_weights = np.array([1 << i for i in range(k)], dtype=int)
    for m in range(1 << n):
        # extrae bits de 'm' en las posiciones 'indices'
        bits = ((m >> np.array(indices)) & 1).astype(int)
        idx = int((bits * bit_weights).sum())
        marg[idx] += full_dist[m]
    total = marg.sum()
    return marg/total if total > 0 else marg

class GeometricSIA(SIA):
    """
    Estrategia geométrica optimizada según el artículo:
      1) BFS por niveles de Hamming + memorización de T_v(i,j) con simetrías
      2) Heurística set‐cover sobre zonas de bajo coste multivariable
      3) Evaluación por suma de EMD en marginals parciales
    """

    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        # Métrica EMD u otra que configure el gestor
        self.distancia = seleccionar_metrica(aplicacion.distancia_metrica)

    def aplicar_estrategia(self, condicion: str, alcance: str, mecanismo: str) -> Solution:
        t0 = time.time()
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        S = self.sia_subsistema
        n = len(S.indices_ncubos)

        # 1) Obtener tensores elementales
        tensores = {nc.indice: nc.data.flatten() for nc in S.ncubos}

        # 2) Construir tabla T[v][(i,j)]
        T = self._construir_tabla_costes(tensores, n)

        # 3) Generar candidatos (set-cover multivariable)
        candidatos = self._generar_candidatos_multivariable(T, n)

        # 4) Evaluar candidatos por EMD en parciales
        dist_full = S.distribucion_marginal()
        best_score = float('inf')
        best_part = None
        best_dist = None

        for mech, alc in candidatos:
            part = S.bipartir(np.array(alc,dtype=int), np.array(mech,dtype=int))
            # marginals parciales
            mS_mech = marginalizar_subconjunto(dist_full, mech)
            mS_alc  = marginalizar_subconjunto(dist_full, alc)
            distP    = part.distribucion_marginal()
            mP_mech = marginalizar_subconjunto(distP, mech)
            mP_alc  = marginalizar_subconjunto(distP, alc)
            score = self.distancia(mP_mech, mS_mech) + self.distancia(mP_alc, mS_alc)
            if score < best_score:
                best_score, best_part = score, (mech, alc)
                best_dist = distP

        if best_part is None:
            raise ValueError("No se encontró bipartición óptima.")

        # formatear partición y dual
        mech, alc = best_part
        idx_pres = set(S.dims_ncubos.data)
        idx_futu = set(S.indices_ncubos.data)
        dual_mech = list(idx_pres - set(mech))
        dual_alc  = list(idx_futu - set(alc))
        fmt = fmt_biparticion([mech, alc], [dual_mech, dual_alc])

        return Solution(
            estrategia="GeometricSIA",
            perdida=best_score,
            distribucion_subsistema=dist_full,
            distribucion_particion=best_dist,
            tiempo_total=time.time()-t0,
            particion=fmt
        )

    def _construir_tabla_costes(
        self,
        tensores: Dict[int,np.ndarray],
        n: int
    ) -> Dict[int, Dict[Tuple[int,int], float]]:
        """
        Para cada variable v construye T[v][(i,j)] = t_v(i,j) usando BFS
        por niveles de Hamming desde cada i, memorizando y aplicando simetría
        por complemento de bits.
        """
        size = 1 << n
        T: Dict[int, Dict[Tuple[int,int], float]] = {}
        for v, tensor in tensores.items():
            Tv: Dict[Tuple[int,int], float] = {}
            memo: Dict[Tuple[int,int], float] = {}
            # Pre‐memoizar auto‐coste
            for i in range(size):
                memo[(i,i)] = 0.0
                Tv[(i,i)]   = 0.0

            # BFS por niveles de Hamming
            for start in range(size):
                # cola con (estado, coste_acum, nivel_Hamming)
                queue = deque([(start, 0.0, 0)])
                visited = {start}
                while queue:
                    state, c_acc, d = queue.popleft()
                    # vecinos a distancia Hamming=d+1
                    for bit in range(n):
                        neigh = state ^ (1 << bit)
                        if neigh in visited:
                            continue
                        # coste local
                        gamma = 2.0 ** (-(d+1))
                        c_edge = gamma * abs(tensor[state] - tensor[neigh])
                        new_cost = c_acc + c_edge
                        # clave canónica bajo complemento de bits
                        key = tuple(sorted((start, neigh)))
                        # actualizar memo y Tv
                        if key not in memo or new_cost < memo[key]:
                            memo[key] = new_cost
                            memo[(key[1], key[0])] = new_cost
                            Tv[(start, neigh)] = new_cost
                            Tv[(neigh, start)] = new_cost
                            queue.append((neigh, new_cost, d+1))
                            visited.add(neigh)
            T[v] = Tv
        return T

    def _generar_candidatos_multivariable(
        self,
        T: Dict[int, Dict[Tuple[int,int], float]],
        n: int
    ) -> List[Tuple[List[int],List[int]]]:
        """
        Heurística de set-cover:
          - Definimos para cada v la 'zona baja' Z_v = {j | existe i: T[v][(i,j)] <= thr_v}
          - Iteramos combinaciones de variables de tamaño k=2..k_max, 
            buscando cubrir la mayor parte de {0..2^n-1} con la intersección 
            de sus Z_v y generamos particiones a partir de unos pocos j representativos.
        """
        # umbral dinámico por variable: ej. percentil 5% en su distribución de costes
        zonas: Dict[int, Set[int]] = {}
        for v, costes in T.items():
            arr = np.fromiter(costes.values(), float)
            thr = np.percentile(arr, 5)  # 5° percentil
            zonas[v] = {j for (i,j),c in costes.items() if c <= thr}

        candidatos: List[Tuple[List[int],List[int]]] = []
        vars_list = list(zonas)
        k_max = min(4, len(vars_list))  # hasta 4 variables combinado
        universo = set(range(1<<n))
        for k in range(2, k_max+1):
            for combo in combinations(vars_list, k):
                inter = set.intersection(*(zonas[v] for v in combo))
                # si inter no vacío, tomamos unos cuantos j aleatorios
                for j in list(inter)[:5]:
                    bits = [(j>>b)&1 for b in range(n)]
                    mech = [i for i,bit in enumerate(bits) if bit==0]
                    alc  = [i for i,bit in enumerate(bits) if bit==1]
                    if mech and alc:
                        candidatos.append((mech,alc))
        # eliminar duplicados manteniendo orden
        seen = set()
        uniq = []
        for mech,alc in candidatos:
            key = (tuple(mech), tuple(alc))
            if key not in seen:
                seen.add(key)
                uniq.append((mech,alc))
        return uniq
