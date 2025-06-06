from src.controllers.manager import Manager

from src.strategies.force import BruteForce
from src.strategies.phi import Phi


def iniciar():
    """Punto de entrada principal"""
                    # ABCD #
    estado_inicial = "1000"
    condiciones =    "1110"
    alcance =        "1110"
    mecanismo =      "1110"

    gestor_sistema = Manager(estado_inicial)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_bf = Phi(gestor_sistema)

    sia_cero = analizador_bf.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )

    print(sia_cero)
