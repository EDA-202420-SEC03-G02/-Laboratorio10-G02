import pytest


def setup_tests():
    empty_graph = new_graph()
    some_graph = new_graph()

    ed_1_2 = new_edge(1, 2, 3.0)
    ed_2_1 = new_edge(2, 1, 3.0)

    edges_1 = new_list()
    add_last(edges_1, ed_1_2)
    edges_2 = new_list()
    add_last(edges_2, ed_2_1)

    put(some_graph["vertices"], 1, edges_1)
    put(some_graph["information"], 1, {"name": "A"})
    put(some_graph["vertices"], 2, edges_2)
    put(some_graph["information"], 2, {"name": "B"})

    some_graph["edges"] = 1

    return empty_graph, some_graph



def test_new_graph():
    graph = new_graph(10, False)
    assert graph["edges"] == 0
    assert graph["in_degree"] == None
    assert graph["vertices"]["capacity"] == new_map(10, 0.5)["capacity"]
    assert graph["vertices"]["type"] == "PROBING"
    assert graph["information"]["capacity"] == new_map(10, 0.5)["capacity"]
    assert graph["information"]["type"] == "PROBING"



def test_insert_vertex():
    empty_graph, some_graph = setup_tests()

    insert_vertex(empty_graph, 1, {"name": "A"})

    assert empty_graph["vertices"]["size"] == 1
    assert empty_graph["information"]["size"] == 1

    insert_vertex(some_graph, 3, {"name": "C"})
    assert some_graph["vertices"]["size"] == 3
    assert some_graph["information"]["size"] == 3

    insert_vertex(some_graph, 1, {"name": "D"})
    assert some_graph["vertices"]["size"] == 3
    assert some_graph["information"]["size"] == 3
    pass



def test_num_vertices():
    empty_graph, some_graph = setup_tests()

    assert num_vertices(empty_graph) == 0
    assert num_vertices(some_graph) == 2



def test_num_edges():
    empty_graph, some_graph = setup_tests()

    assert num_edges(empty_graph) == 0

    assert num_edges(some_graph) == 1



def test_vertices():
    empty_graph, some_graph = setup_tests()

    verticess = vertices(empty_graph)

    
    assert len(verticess) == 0



def test_edges():
    empty_graph, some_graph = setup_tests()

    edgess = edges(empty_graph)

    assert size(edgess) == 0

    edgess = edges(some_graph)

    assert size(edgess) == 1
    assert edgess["elements"] is not None



def test_degree():
    empty_graph, some_graph = setup_tests()

    assert degree(empty_graph, 1) is None

    assert degree(some_graph, 0) is None
    assert degree(some_graph, 1) == 1
    assert degree(some_graph, 2) == 1



def test_in_degree():
    empty_graph, some_graph = setup_tests()

    assert in_degree(empty_graph, 1) is None
    assert in_degree(some_graph, 0) is None
    assert in_degree(some_graph, 1) is None



def test_add_edge():
    empty_graph, some_graph = setup_tests()

    add_edge(empty_graph, 1, 2, 3.0)

    assert num_edges(empty_graph) == 0

    insert_vertex(some_graph, 3, {"name": "D"})
    add_edge(some_graph, 1, 3, 3.0)

    assert num_edges(some_graph) == 2


#funciones 
def new_graph(size=19, directed=False):
    """
    Crea un grafo vacío con los atributos necesarios para manejar vértices y aristas.
    """
    graph = {
        "vertices": new_map(size, 0.5),  # Tabla hash para vértices
        "information": new_map(size, 0.5),  # Tabla hash para información de vértices
        "in_degree": new_map(size, 0.5) if directed else None,  # Tabla hash para grados de entrada
        "edges": 0,  # Contador de aristas
        "directed": directed,  # Indica si es dirigido
        "type": "ADJ_LIST"  # Tipo del grafo
    }
    return graph


def insert_vertex(graph, key_vertex, info_vertex=None):
    """
    Inserta un vértice al grafo. Si ya existe, actualiza la información asociada.
    """
    if not contains(graph["vertices"], key_vertex):
        put(graph["vertices"], key_vertex, new_list())  # Lista vacía de adyacencia
        put(graph["information"], key_vertex, info_vertex)
        if graph["directed"] and graph["in_degree"] is not None:
            put(graph["in_degree"], key_vertex, 0)  # Inicializa grado de entrada
    else:
        put(graph["information"], key_vertex, info_vertex)  # Actualiza información


def add_edge(graph, vertex_a, vertex_b, weight=0):
    """
    Agrega un arco al grafo. Si ya existe, actualiza el peso. Si no existe, lo crea.
    """
    # Verifica si el grafo está vacío
    if graph["edges"] == 0 and not contains(graph["vertices"], vertex_a) and not contains(graph["vertices"], vertex_b):
        return  # No hacer nada si el grafo está vacío y no hay vértices

    if not contains(graph["vertices"], vertex_a):
        insert_vertex(graph, vertex_a)
    if not contains(graph["vertices"], vertex_b):
        insert_vertex(graph, vertex_b)

    edges_a = get(graph["vertices"], vertex_a)
    edges_b = get(graph["vertices"], vertex_b)

    # Verifica si el arco ya existe y actualiza el peso si es necesario
    for edge in edges_a["elements"]:
        if edge["vertex_b"] == vertex_b:
            set_weight(edge, weight)
            return

    # Agrega el nuevo arco
    add_last(edges_a, new_edge(vertex_a, vertex_b, weight))
    if graph["directed"]:
        in_degree_value = get(graph["in_degree"], vertex_b)
        put(graph["in_degree"], vertex_b, in_degree_value + 1)
    else:
        add_last(edges_b, new_edge(vertex_b, vertex_a, weight))

    graph["edges"] += 1


def num_vertices(graph):
    """Retorna el número de vértices en el grafo."""
    return size(graph["vertices"])


def num_edges(graph):
    """Retorna el número de arcos en el grafo."""
    return graph["edges"]


def vertices(graph):
    """Retorna una lista con todos los vértices del grafo."""
    keys = get_keys(graph["vertices"])  # Obtener todas las llaves de la tabla hash
    return keys


def edges(graph):
    """
    Retorna una lista con todos los arcos del grafo.

    :param graph: Grafo del que se obtienen los arcos.
    :type graph: dict
    :return: Lista de arcos en el grafo.
    :rtype: list
    """
    edge_list = new_list()
    visited = set()  # Para evitar duplicados en grafos no dirigidos

    for vertex in vertices(graph):
        adj_list = get(graph["vertices"], vertex)
        for edge in adj_list["elements"]:
            vertex_b = edge["vertex_b"]
            if graph["directed"] or (vertex, vertex_b) not in visited:
                add_last(edge_list, edge)
                if not graph["directed"]:
                    visited.add((vertex_b, vertex))

    return edge_list


def degree(graph, key_vertex):
    """
    Retorna el número de arcos asociados al vértice con llave `key_vertex`.
    Retorna None si el vértice no existe.
    """
    if not contains(graph["vertices"], key_vertex):
        return None
    adj_list = get(graph["vertices"], key_vertex)
    return size(adj_list)


def in_degree(graph, key_vertex):
    """
    Retorna el número de arcos que llegan al vértice con llave `key_vertex`.
    Retorna None si el vértice no existe.
    """
    if not graph["directed"] or not contains(graph["vertices"], key_vertex):
        return None

    return get(graph["in_degree"], key_vertex)











#funciones adicionales
def new_list():
    return {
        "elements": [],
        "size": 0
    }

def get_list(catalog, key):
    return catalog.get(key, [])     

def add_first(lst, elem):
    if lst is None:
        raise ValueError("List cannot be None")
    lst["elements"].insert(0, elem)
    lst["size"] += 1
    return lst

def add_last(lst, elem):
    if lst is None:
        raise ValueError("List cannot be None")
    lst["elements"].append(elem)
    lst["size"] += 1
    return lst

def add_all(lst, elems):
    if lst is None:
        raise ValueError("List cannot be None")
    for elem in elems:
        add_last(lst, elem)
    return lst

def is_empty(lst):
    if lst is None:
        raise ValueError("List cannot be None")
    return lst["size"] == 0

def size(lst):
    if lst is None:
        raise ValueError("List cannot be None")
    return lst["size"]

def get_first_element(lst):
    if lst is None:
        raise ValueError("List cannot be None")
    if lst["size"] == 0:
        return None
    return lst["elements"][0]

def get_last_element(lst):
    if lst is None:
        raise ValueError("List cannot be None")
    if lst["size"] == 0:
        return None
    return lst["elements"][-1]

def get_element(lst, pos):
    if lst is None:
        raise ValueError("List cannot be None")
    if 0 <= pos < lst["size"]:
        return lst["elements"][pos]
    return None

def remove_first(lst):
    if lst is None:
        raise ValueError("List cannot be None")
    if lst["size"] > 0:
        first_element = lst["elements"].pop(0)
        lst["size"] -= 1
        return first_element
    return None

def remove_last(lst):
    if lst is None:
        raise ValueError("List cannot be None")
    if lst["size"] == 0:
        return None
    last_element = lst["elements"].pop()
    lst["size"] -= 1
    return last_element

def insert_element(lst, elem, pos):
    if lst is None:
        raise ValueError("List cannot be None")
    if 0 <= pos <= lst["size"]:
        lst["elements"].insert(pos, elem)
        lst["size"] += 1
    return lst

def is_present(lst, elem, cmp_function):
    if lst is None:
        raise ValueError("List cannot be None")
    for keypos, info in enumerate(lst["elements"]):
        if cmp_function(elem, info):  # Ajuste aquí
            return keypos
    return -1

def delete_element(lst, pos):
    if lst is None:
        raise ValueError("List cannot be None")
    if 0 <= pos < lst["size"]:
        lst["elements"].pop(pos)
        lst["size"] -= 1
        return lst
    return None

def change_info(lst, pos, new_info):
    if lst is None:
        raise ValueError("List cannot be None")
    if 0 <= pos < lst["size"]:
        lst["elements"][pos] = new_info
    return lst

def exchange(lst, pos1, pos2):
    if lst is None:
        raise ValueError("List cannot be None")
    if 0 <= pos1 < lst["size"] and 0 <= pos2 < lst["size"]:
        lst["elements"][pos1], lst["elements"][pos2] = lst["elements"][pos2], lst["elements"][pos1]
    return lst

def sub_list(lst, pos, numelem):
    if lst is None:
        raise ValueError("List cannot be None")
    if 0 <= pos < lst["size"] and 0 <= numelem <= lst["size"] - pos:
        copia = lst["elements"][pos:pos + numelem]
        sub_lst = {
            "elements": copia,
            "size": len(copia)
        }
        return sub_lst
    return None

#sort



import random

# TODO selection sort completar array: EST-2
def selection_sort(lst, sort_crit):
    size = lst['size']
    elements = lst['elements']

    for i in range(size):
        minimo = i  # Suponemos que el primer elemento no ordenado es el mínimo
        for j in range(i + 1, size):
            # Usamos la función de criterio de ordenación para comparar
            if sort_crit(elements[j], elements[minimo]):
                minimo = j  # Actualizamos el índice del mínimo si encontramos uno más pequeño
        # Intercambiamos los elementos
        elements[i], elements[minimo] = elements[minimo], elements[i]

    return lst  # Retornamos el diccionario con la lista ordenada


# TODO insertion sort completar array: EST-2
def insertion_sort(lst, sort_crit):
    size = lst['size']
    elements = lst['elements']

    if size <= 1:  # Si la longitud de la lista es 0 o 1, ya está ordenada
        return lst
    for i in range(1, size):
        current_element = elements[i]  # Elemento actual a insertar
        j = i - 1  # Índice para recorrer la parte ordenada hacia atrás
        
        # Desplazamos los elementos de la parte ordenada hacia la derecha
        # mientras sean mayores que el elemento actual
        while j >= 0 and sort_crit(current_element, elements[j]):
            elements[j + 1] = elements[j]  # Desplazar hacia la derecha
            j -= 1  # Mover hacia el inicio de la lista
        
        # Colocar el elemento actual en la posición correcta
        elements[j + 1] = current_element

    return lst  # Retornar el diccionario con la lista ordenada


# TODO shell sort completar array: EST-2
def shell_sort(lst, sort_crit):
    size = lst['size']
    elements = lst['elements']

    if size <= 1:  # Si la lista tiene un solo elemento o está vacía, no necesita ser ordenada.
        return lst

    gap = size // 2  # Inicializamos el hueco (gap) en la mitad de la longitud de la lista.

    while gap > 0:
        for i in range(gap, size):
            current_element = elements[i]  # Guardamos el elemento actual que queremos insertar.
            j = i  # Inicializamos j en la posición actual i para comenzar las comparaciones.

            # Movemos los elementos mayores que current_element hacia la derecha.
            while j >= gap and sort_crit(current_element, elements[j - gap]):
                elements[j] = elements[j - gap]  # Desplazamos el elemento hacia la derecha.
                j -= gap  # Decrementamos j para continuar verificando elementos en el hueco.

            # Insertamos current_element en su posición correcta.
            elements[j] = current_element

        # Reducimos el gap a la mitad para la próxima iteración.
        gap //= 2  

    return lst  # Devolvemos el diccionario con la lista ordenada.


# TODO merge sort completar array: EST-2
def merge_sort(lst, sort_crit):
    size = lst['size']
    elements = lst['elements']

    if size <= 1:  # Si la lista tiene un solo elemento o está vacía, ya está ordenada
        return lst

    # Divide la lista en dos mitades
    mid = size // 2  # Encuentra el punto medio
    left_half = {'size': mid, 'elements': elements[0:mid]}  # Crea la mitad izquierda
    right_half = {'size': size - mid, 'elements': elements[mid:]}  # Crea la mitad derecha

    # Llama recursivamente a la mitad izquierda y derecha
    left_half = merge_sort(left_half, sort_crit)
    right_half = merge_sort(right_half, sort_crit)

    # Fusiona las dos mitades ordenadas
    merged = merge(left_half, right_half, sort_crit)

    # Actualiza la lista original con los elementos ordenados
    lst['elements'] = merged['elements']
    return lst


def merge(left, right, sort_crit):
    # Inicializa una lista para almacenar los elementos fusionados
    merged = [0] * (left['size'] + right['size'])  # Crea una lista del tamaño combinado
    left_index = right_index = merged_index = 0  # Índices para recorrer ambas listas y la lista fusionada

    # Compara elementos de ambas listas y los agrega a la lista fusionada
    while left_index < left['size'] and right_index < right['size']:
        if sort_crit(left['elements'][left_index], right['elements'][right_index]):
            merged[merged_index] = left['elements'][left_index]  # Agrega el elemento de la izquierda
            left_index += 1  # Avanza el índice de la izquierda
        else:
            merged[merged_index] = right['elements'][right_index]  # Agrega el elemento de la derecha
            right_index += 1  # Avanza el índice de la derecha
        merged_index += 1  # Avanza el índice de la lista fusionada

    # Si quedan elementos en la lista izquierda, agréguelos a la lista fusionada
    while left_index < left['size']:
        merged[merged_index] = left['elements'][left_index]
        left_index += 1
        merged_index += 1

    # Si quedan elementos en la lista derecha, agréguelos a la lista fusionada
    while right_index < right['size']:
        merged[merged_index] = right['elements'][right_index]
        right_index += 1
        merged_index += 1

    return {'size': len(merged), 'elements': merged} 


# TODO quick sort completar array: EST-1
def quick_sort(lst, sort_crit):
    quick_sort_recursive(lst, 0, lst['size'] - 1, sort_crit)

# Función recursiva para Quick Sort
def quick_sort_recursive(lst, lo, hi, sort_crit):
    if lo < hi:  # Solo particionar si la lista tiene más de un elemento
        p = partition(lst, lo, hi, sort_crit)  # Particiona la lista y encuentra la posición del pivote
        quick_sort_recursive(lst, lo, p - 1, sort_crit)  # Ordena recursivamente la sublista izquierda
        quick_sort_recursive(lst, p + 1, hi, sort_crit)  # Ordena recursivamente la sublista derecha

# Función de partición
def partition(lst, lo, hi, sort_crit):
    index_piv = random.randrange(lo, hi + 1)  # Selecciona un pivote aleatorio
    lst['elements'][index_piv], lst['elements'][hi] = lst['elements'][hi], lst['elements'][index_piv]  # Intercambiar el pivote aleatorio con el último elemento
    pivot = lst['elements'][hi]  # Pivote es el último elemento
    i = lo - 1  # Índice del menor elemento
    
    for j in range(lo, hi):  
        if sort_crit(lst['elements'][j], pivot):  # Si el elemento es menor o igual que el pivote
            i += 1  # Incrementa el índice del menor elemento
            lst['elements'][i], lst['elements'][j] = lst['elements'][j], lst['elements'][i]  # Intercambia lst[i] con lst[j]
    
    lst['elements'][i + 1], lst['elements'][hi] = lst['elements'][hi], lst['elements'][i + 1]  # Coloca el pivote en su posición final
    return i + 1  # Devuelve el índice final del pivote

# Función de comparación por defecto para ordenar de manera ascendente
def default_sort_criteria(element1, element2):
    return element1 <= element2
#lp
import random



def new_map(num_elements, load_factor):
    prime=109345121
    # capacity: Tamaño de la tabla. Siguiente número primo mayor a num_elements/load_factor
    capacity = next_prime(int(num_elements / load_factor))
    # scale: Número aleatorio entre 1 y prime-1
    scale = random.randint(1, prime - 1)
    # shift: Número aleatorio entre 0 y prime-1
    shift = random.randint(0, prime - 1)
    # table: Lista de tamaño capacity con las entradas de la tabla
    table = [None] * capacity   
    # Crear el mapa con los atributos iniciales
    nuevo_mapa = {
        'prime': prime,
        'capacity': capacity,
        'scale': scale,
        'shift': shift,
        'table': table,
        'current_factor': 0,  # inicialmente no hay elementos
        'limit_factor': load_factor,
        'size': 0,  # no hay elementos en la tabla inicialmente
        'type': 'PROBING'  # usando sondeo lineal
    }
    
    return nuevo_mapa


import random



def put(my_map, key, value):
    """
    Inserta un par llave-valor en la tabla hash.
    Redimensiona la tabla si el factor de carga excede el límite.

    :param my_map: La tabla hash donde insertar.
    :type my_map: dict
    :param key: La llave a insertar.
    :type key: any
    :param value: El valor asociado a la llave.
    :type value: any
    """
    if my_map["current_factor"] >= my_map["limit_factor"]:
        resize(my_map)  # Redimensionar la tabla

    index = hash_value(my_map, key)
    while my_map["table"][index] is not None:
        existing_key, _ = my_map["table"][index]
        if existing_key == key:
            # Actualizar valor si la llave ya existe
            my_map["table"][index] = (key, value)
            return
        index = (index + 1) % my_map["capacity"]  # Sondeo lineal

    # Insertar nueva entrada
    my_map["table"][index] = (key, value)
    my_map["size"] += 1
    my_map["current_factor"] = my_map["size"] / my_map["capacity"]
def resize(map_):
    """
    Duplica el tamaño de la tabla hash y reubica los elementos.

    :param map_: La tabla hash a redimensionar.
    :type map_: dict
    """
    old_table = map_["table"]
    new_capacity = next_prime(map_["capacity"] * 2)
    map_["table"] = [None] * new_capacity
    map_["capacity"] = new_capacity
    map_["size"] = 0

    for entry in old_table:
        if entry is not None:
            key, value = entry
            put(map_, key, value)

def contains(my_map, key):
    """Valida si la llave key se encuentra en el my_map."""
    index = hash_value(my_map, key)  # Calcula el índice usando hash_value
    table = my_map['table']
    capacity = my_map['capacity']

    # Usar sondeo lineal para buscar la llave
    initial_index = index
    while table[index] is not None:
        stored_key, _ = table[index]

        # Si encontramos la llave, retornamos True
        if stored_key == key:
            return True

        # Sondeo lineal: probamos la siguiente posición
        index = (index + 1) % capacity

        # Verificación para evitar un ciclo infinito
        if index == initial_index:
            break

    # Si no encontramos la llave, retornamos False
    return False

def get_keys(my_map):
    """ 
    Retorna todas las llaves en el mapa.
    """
    keys = []
    for entry in my_map['table']:
        if entry is not None:  # Verifica si la entrada no está vacía
            keys.append(entry[0])  # Agrega la clave a la lista de claves
    return keys

def get(my_map, key):
    """Recupera el valor asociado a la llave en el mapa de hash."""
    index = hash_value(my_map, key)  # Calcula el índice usando hash_value
    table = my_map['table']
    capacity = my_map['capacity']

    # Usar sondeo lineal para buscar la llave
    initial_index = index
    while table[index] is not None:
        stored_key, stored_value = table[index]

        # Si encontramos la llave, retornamos el valor asociado
        if stored_key == key:
            return stored_value

        # Sondeo lineal: probamos la siguiente posición
        index = (index + 1) % capacity

        # Verificación para evitar un ciclo infinito
        if index == initial_index:
            break

    # Si no encontramos la llave, retornamos None
    return None


def remove(my_map, key):
    """Elimina una pareja llave-valor del mapa."""
    index = hash_value(my_map, key)
    table = my_map['table']
    capacity = my_map['capacity']
    
    # Usa sondeo lineal para buscar la llave
    initial_index = index
    while table[index] is not None:
        stored_key, _ = table[index]
        if stored_key == key:
            table[index] = None  # Marca como vacío
            my_map['size'] -= 1
            return
        index = (index + 1) % capacity
        
        # Verificación para evitar un ciclo infinito
        if index == initial_index:
            break  # Salir si volvemos al índice inicial


def size(my_map):
    """Retorna el número de parejas llave-valor en el mapa."""
    return my_map['size']


def is_empty(my_map):
    """Indica si el mapa está vacío."""
    return not my_map['size']


def key_set(my_map):
    """Retorna una lista con todas las llaves de la tabla de hash."""
    keys = []
    table = my_map['table']

    # Recorrer la tabla y extraer las claves
    for entry in table:
        if entry is not None:
            key, _ = entry
            keys.append(key)

    return keys


def value_set(my_map):
    """Retorna una lista con todos los valores de la tabla de hash."""
    values = []
    table = my_map['table']

    # Recorrer la tabla y extraer los valores
    for entry in table:
        if entry is not None:
            _, value = entry
            values.append(value)

    return values


def find_slot(my_map, key, hash_value):
    """Busca la llave a partir de una posición dada en la tabla."""
    table = my_map['table']
    capacity = my_map['capacity']
    index = hash_value % capacity

    while True:
        if table[index] is None:
            return False, index  # Regresar que no está ocupada
        if table[index] == "__EMPTY__":
            index = (index + 1) % capacity
            continue
        
        stored_key, _ = table[index]
        if stored_key == key:
            return True, index
        index = (index + 1) % capacity   

def is_available(table, pos):
    """Informa si la posición está disponible en la tabla de hash."""
    return table[pos] is None or table[pos] == '__EMPTY__'


def rehash(my_map):
    """Hace rehash de todos los elementos de la tabla de hash."""
    old_table = my_map['table']
    new_capacity = next_prime(my_map['capacity'] * 2)
    my_map['capacity'] = new_capacity
    my_map['table'] = [None] * new_capacity
    my_map['size'] = 0  # Reseteamos el tamaño

    # Reubicar todas las entradas que no sean None ni __EMPTY__
    for entry in old_table:
        if entry is not None and entry != '__EMPTY__':
            key, value = entry
            put(my_map, key, value)  # Reinsertar en la nueva tabla con el tamaño ajustado
    
    return my_map


def default_compare(key, element):
    """Función de comparación por defecto."""
    if isinstance(element, tuple):
        if key == element[0]:
            return 0
        elif key > element[0]:
            return 1
        else:
            return -1
    else:
        if key == element:
            return 0
        elif key > element:
            return 1
        else:
            return -1
    



import math

"""
    Funciones auxiliares para el manejo de tablas de simbolos (**mapas**)
"""

def is_prime(n):
    """ Valida si un número es primo o no

        :param n: Número a validar
        :type n: int

        :return: True si es primo, False en caso contrario
    """
    # Corner cases
    if(n <= 1):
        return False
    if(n <= 3):
        return True

    if(n % 2 == 0 or n % 3 == 0):
        return False

    for i in range(5, int(math.sqrt(n) + 1), 6):
        if(n % i == 0 or n % (i + 2) == 0):
            return False

    return True

def next_prime(n):
    """ Encuentra el siguiente número primo mayor a n

        :param n: Número a partir del cual se busca el siguiente primo
        :type n: int

        :return: El siguiente número primo mayor a n
    """
    found = False
    next_p = 1
    # Base case
    if (n <= 1):
        next_p = 2
        found = True
    next_p = int(n)
    # Loop continuously until is_prime returns
    # True for a number greater than n
    while(not found):
        next_p = next_p + 1
        if is_prime(next_p):
            found = True
    return int(next_p)

def hash_value(table, key):
    """
    Calcula un hash para una llave utilizando el método MAD.

    :param table: Tabla hash inicializada con los parámetros requeridos.
    :type table: dict
    :param key: Llave a la cual se calculará el hash.
    :type key: any
    :return: Valor hash para la llave.
    :rtype: int
    """
    p = table["prime"]
    a = table["scale"]
    b = table["shift"]
    M = table["capacity"]  # Cambiado de "size" a "capacity" para evitar divisiones por 0

    if M == 0:
        raise ValueError("La capacidad de la tabla no puede ser 0.")

    h = (hash(key) * a + b) % p  # Método MAD
    return h % M


def new_my_mapentry(key, value):
    """ Retorna una pareja llave valor para ser guardada en un Map

        :param key: Llave de la pareja
        :type key: any
        :param value: Valor de la pareja
        :type value: any

        :return: Una entrada con la pareja llave-valor
        :rtype: my_mapentry
    """
    entry = {'key': key, 'value': value}
    return entry


def set_key(my_entry, key):
    """ Asigna un valor nuevo a la ``key`` del entry recibido como parámetro

        :param my_entry: La pareja llave-valor
        :type my_entry: my_mapentry
        :param key: La nueva llave
        :type key: any

        :return: La pareja modificada
        :rtype: my_mapentry
    """
    my_entry['key'] = key
    return my_entry


def set_value(my_entry, value):
    """Asigna un valor nuevo al ``value`` del entry recibido como parámetro

        :param my_entry: La pareja llave-valor
        :type my_entry: my_mapentry
        :param value: El nuevo value
        :type value: any

        :return: La pareja modificada
        :rtype: my_mapentry
    """
    my_entry['value'] = value
    return my_entry


def get_key(my_entry):
    """ 
    Retorna la llave de la entry recibida como parámetro

    :param my_entry: La pareja llave-valor
    :type my_entry: my_mapentry

    :return: La llave de la pareja
    :rtype: any
    """
    return my_entry['key']


def get_value(my_entry):
    """
    Retorna el valor de la entry recibida como parámetro

    :param my_entry: La pareja llave-valor
    :type my_entry: my_mapentry
    
    :return: El valor de la pareja
    :rtype: any
    """
    return my_entry['value']
def initialize_table(size, prime=None, scale=None, shift=None):
    """
    Inicializa una tabla hash con los parámetros requeridos para el método MAD.

    :param size: Tamaño de la tabla.
    :type size: int
    :param prime: Número primo mayor al tamaño de la tabla (opcional).
    :type prime: int
    :param scale: Factor de escala (opcional).
    :type scale: int
    :param shift: Desplazamiento (opcional).
    :type shift: int

    :return: Tabla hash inicializada.
    :rtype: dict
    """
    import random

    if prime is None:
        prime = next_prime(size * 2)  # Asegúrate de calcular un primo mayor al tamaño
    if scale is None:
        scale = random.randint(1, prime - 1)  # 1 <= scale < prime
    if shift is None:
        shift = random.randint(0, prime - 1)  # 0 <= shift < prime

    return {
        "size": size,
        "prime": prime,
        "scale": scale,
        "shift": shift,
        "buckets": {i: [] for i in range(size)},  # Inicializa los buckets
    }

def next_prime(n):
    """
    Encuentra el siguiente número primo mayor o igual a n.

    :param n: Número inicial.
    :type n: int

    :return: El siguiente número primo.
    :rtype: int
    """
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True

    while not is_prime(n):
        n += 1
    return n
def new_edge(v_a, v_b, weight=0):
    """
    Crea un nuevo arco entrelos vertices ``v_a`` y ``v_b`` con un peso ``weight``

    Se crea un arco con los siguientes atributos:

    - **vertex_a**: Vertice A del arco
    - **vertex_b**: Vertice B del arco
    - **weight**: Peso del arco

    :param v_a: Vertice A del arco
    :type v_a: any
    :param v_b: Vertice B del arco
    :type v_b: any
    :param weight: Peso del arco
    :type weight: double

    :returns: Arco creado
    :rtype: edge
    """
    edge = {"vertex_a": v_a, "vertex_b": v_b, "weight": weight}
    return edge


def weight(edge):
    """
    Retorna el peso del arco ``edge``

    :param edge: Arco del cual se quiere obtener el peso
    :type edge: edge

    :returns: Peso del arco
    :rtype: double
    """
    return edge["weight"]


def either(edge):
    """
    Retorna el vertice A del arco ``edge``

    :param edge: Arco del cual se quiere obtener el vertice A
    :type edge: edge

    :returns: Vertice A del arco
    :rtype: any
    """
    return edge["vertex_a"]


def other(edge, veither):
    """
    Retorna el vertice del arco ``edge`` que no es igual a ``veither``

    :param edge: Arco del cual se quiere obtener el vertice B
    :type edge: edge
    :param veither: Vertice A del arco
    :type veither: any

    :returns: Vertice B del arco
    :rtype: any
    """
    if veither == edge["vertex_a"]:
        return edge["vertex_b"]
    elif veither == edge["vertex_b"]:
        return edge["vertex_a"]


def set_weight(edge, weight):
    """
    Cambia el peso del arco ``edge`` por el valor ``weight``

    :param edge: Arco al cual se le quiere cambiar el peso
    :type edge: edge
    :param weight: Nuevo peso del arco
    :type weight: double
    """
    edge["weight"] = weight


def compare_edges(edge1, edge2):
    """
    Funcion utilizada en lista de edges para comparar dos edges
    Retorna 0 si los arcos son iguales, 1 si edge1 > edge2, -1 edge1 < edge2

    :param edge1: Arco 1
    :type edge1: edge
    :param edge2: Arco 2
    :type edge2: edge

    :returns: 0 si los arcos son iguales, 1 si edge1 > edge2, -1 edge1 < edge2
    :rtype: int
    """
    e1v = either(edge1)
    e2v = either(edge2)

    if e1v == e2v:
        if other(edge1, e1v) == other(edge2, e2v):
            return 0
        elif other(edge1, e1v) > other(edge2, e2v):
            return 1
        else:
            return -1
    elif e1v > e2v:
        return 1
    else:
        return -1