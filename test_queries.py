"""Dos comprobaciones manuales: respuesta presente y pregunta trampa."""

import asyncio

from rag import get_rag_response


async def run_checks() -> None:
    known = await get_rag_response("¿Cada cuanto debo revisar la humedad del sustrato?")
    assert known.answer != "No lo sé", "La pregunta cubierta deberia tener respuesta"
    assert known.references, "La respuesta cubierta deberia incluir referencias"
    print("PRUEBA 1 - informacion presente")
    print(known.model_dump_json(indent=2))

    trap = await get_rag_response("¿Quien gano el Mundial de futbol de 1998?")
    assert trap.answer == "No lo sé", "La pregunta trampa no debe ser contestada"
    assert trap.references == [], "La pregunta trampa no debe citar fuentes"
    print("\nPRUEBA 2 - pregunta trampa")
    print(trap.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(run_checks())
