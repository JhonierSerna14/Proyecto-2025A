from src.controllers.manager import Manager

from src.strategies.geometric import GeometricSIA
from src.strategies.q_nodes import QNodes


def iniciar():
    """Punto de entrada principal"""
                    # 23 bits
    estado_inicial = "1000000000000000000000000"
    condiciones =    "1111111111111111111111111"
    alcance =        "1111111111111111111111111"
    mecanismo =      "1111111111111111111111111"

    gestor_sistema = Manager(estado_inicial)

    # ✅ Verifica que existe TPM de 23 nodos, o créala si no
    if not gestor_sistema.tpm_filename.exists():
        print(f"Archivo TPM de {len(estado_inicial)} nodos no encontrado. Generando uno nuevo...")
        gestor_sistema.generar_red(dimensiones=len(estado_inicial), datos_discretos=True)

    
    analizador_qn = GeometricSIA(gestor_sistema)
    sia_uno = analizador_qn.aplicar_estrategia(condiciones, alcance, mecanismo)
    print(sia_uno)
    
    