def new_graph(size=19, directed=False):
    """
    Crea un grafo vacío con los atributos necesarios para manejar vértices y aristas.

    El grafo tiene las siguientes características:
    - **vertices**: Mapa de vértices, donde cada vértice apunta a una lista de aristas adyacentes.
    - **information**: Mapa que asocia información a cada vértice (opcional).
    - **in_degree**: Mapa de grados de entrada de los vértices (solo si el grafo es dirigido).
    - **edges**: Número total de aristas en el grafo.
    - **directed**: Indica si el grafo es dirigido o no.
    - **type**: Tipo de grafo (inicializado en "ADJ_LIST").

    :param size: Número inicial de vértices (opcional, por defecto 19).
    :type size: int
    :param directed: Indica si el grafo es dirigido (opcional, por defecto False).
    :type directed: bool

    :returns: Grafo vacío.
    :rtype: dict
    """
    graph = {
        "vertices": {i: [] for i in range(size)},  # Inicializa lista de adyacencia
        "information": {i: {} for i in range(size)},  # Información asociada a cada vértice
        "in_degree": {i: 0 for i in range(size)} if directed else None,  # Grado de entrada para grafos dirigidos
        "edges": 0,  # Contador de aristas
        "directed": directed,  # Indica si el grafo es dirigido
        "type": "ADJ_LIST"  # Tipo de grafo
    }
    return graph



def add_edge(graph, from_vertex, to_vertex):
    if from_vertex in graph["vertices"] and to_vertex in graph["vertices"]:
        graph["vertices"][from_vertex].append(to_vertex)
        if not graph["directed"]:
            graph["vertices"][to_vertex].append(from_vertex)
        graph["edges"] += 1
    else:
        raise ValueError("One or both vertices do not exist in the graph.")