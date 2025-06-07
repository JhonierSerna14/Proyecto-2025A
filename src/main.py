from src.controllers.manager import Manager

from src.strategies.force import BruteForce
from src.strategies.phi import Phi
from src.strategies.geometric import GeometricSIA


def iniciar():
    """Punto de entrada principal"""
                    # ABCD #
    estado_inicial = "100000000000000"
    condiciones =    "111000000000000"
    alcance =        "111000000000000"
    mecanismo =      "111100000000000"

    gestor_sistema = Manager(estado_inicial)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    # analizador_bf = GeometricSIA(gestor_sistema)
    analizador_bf = Phi(gestor_sistema)

    sia_cero = analizador_bf.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )

    print(sia_cero)
