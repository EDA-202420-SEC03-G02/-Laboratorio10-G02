def new_list():
    return {'first': None, 'last': None, 'size': 0}

def add_last(lst, elem):
    nodo = new_single_node(elem)
    if lst["size"] == 0:
        lst["first"] = nodo
        lst["last"] = nodo
    else:
        lst["last"]["next"] = nodo
        lst["last"] = nodo
    lst["size"] += 1
    return lst

def add_first(lst, elem):
    nodo = new_single_node(elem)
    if lst["size"] == 0:
        lst["first"] = nodo
        lst["last"] = nodo
    else:
        nodo["next"] = lst["first"]
        lst["first"] = nodo
    lst["size"] += 1
    return lst

def size(lst):
    return lst["size"]

def get_first_element(lst):
    return lst['first']['info'] if lst['first'] is not None else None

def remove_first(lst):
    if lst["first"] is not None:
        first_elem = lst["first"]["info"]
        lst["first"] = lst["first"]["next"]
        lst["size"] -= 1
        if lst["first"] is None:
            lst["last"] = None
        return first_elem
    return None

def is_empty(lst):
    return lst["size"] == 0

def remove_last(lst):
    if lst["first"] is None:
        return None

    if lst["first"]["next"] is None: 
        data = lst["first"]["info"]
        lst["first"] = lst["last"] = None
        lst["size"] -= 1
        return data
    
    current = lst["first"]
    while current["next"]["next"] is not None:
        current = current["next"]

    data = current["next"]["info"]
    current["next"] = None
    lst["last"] = current
    lst["size"] -= 1
    return data

def get_last_element(lst):
    return lst["last"]["info"] if lst["last"] is not None else None

def get_element(lst, pos):
    if pos < 0 or pos >= lst["size"]:
        return None

    current = lst["first"]
    for index in range(pos):
        current = current["next"]
    
    return current["info"] if current is not None else None

def delete_element(lst, pos):
    if pos < 0 or pos >= lst["size"]:
        return None

    if pos == 0:
        return remove_first(lst)

    current = lst["first"]
    for index in range(pos - 1):
        current = current["next"]

    if current is None or current["next"] is None:
        return None

    d = current["next"]["info"]
    current["next"] = current["next"]["next"]
    if current["next"] is None:
        lst["last"] = current
    lst["size"] -= 1
    return d

def is_present(lst, elem, cmp_function):
    current = lst['first']
    pos = 0
    while current is not None:
        if cmp_function(elem, current['info']) == 0:
            return pos
        current = current['next']
        pos += 1
    return -1

def change_info(lst, pos, new_info):
    current = lst["first"]
    for index in range(pos):
        if current is None:
            return lst
        current = current["next"]

    if current is not None:
        current["info"] = new_info
    return lst 

def exchange(lst, pos1, pos2):
    if pos1 == pos2:
        return lst 

    node1 = node2 = None
    current = lst["first"]
    for index in range(max(pos1, pos2) + 1):
        if current is None:
            break
        if index == pos1:
            node1 = current
        if index == pos2:
            node2 = current
        current = current["next"]

    if node1 is not None and node2 is not None:
        node1["info"], node2["info"] = node2["info"], node1["info"]

    return lst
  
def sub_list(lst, pos, numelem):
    sub_lst = new_list()
    current = lst["first"]

    for index in range(pos):
        if current is None:
            return sub_lst
        current = current["next"]
        
    for _ in range(numelem):
        if current is None:
            break
        add_last(sub_lst, current["info"])
        current = current["next"]

    return sub_lst

def insert_element(lst, elem, pos):
    if pos < 0 or pos > lst["size"]:
        raise IndexError("La posición está fuera de los límites de la lista.")

    new_node = new_single_node(elem)

    if pos == 0:
        new_node["next"] = lst["first"]
        lst["first"] = new_node
        if lst["size"] == 0:  
            lst["last"] = new_node
    else:
        current = lst["first"]
        for index in range(pos - 1):
            current = current["next"]

        new_node["next"] = current["next"]
        current["next"] = new_node

        if new_node["next"] is None:
            lst["last"] = new_node

    lst["size"] += 1
    return lst



#sort

# Función de comparación por defecto para ordenar de manera ascendente.
def default_sort_criteria(element1, element2):
    return element1 < element2


def selection_sort(my_list, sort_crit):
    if my_list['size'] < 2:  # Si la lista tiene menos de 2 elementos, ya está ordenada.
        return my_list

    current = my_list['first']
    while current is not None:
        min_node = current  # Suponemos que el nodo actual es el mínimo
        next_node = current['next']

        
        while next_node is not None:
            if sort_crit(next_node['info'], min_node['info']):
                min_node = next_node
            next_node = next_node['next']

        # Intercambiamos el valor del nodo mínimo con el valor del nodo actual
        current['info'], min_node['info'] = min_node['info'], current['info']
        current = current['next']  # Pasamos al siguiente nodo

    return my_list


def insertion_sort(my_list, sort_crit):
    if my_list['size'] < 2:  # Si la lista tiene menos de 2 elementos, ya está ordenada.
        return my_list

    sorted_list = new_list()  # Creamos una nueva lista ordenada
    current = my_list['first']

    while current is not None:
        # Insertamos el elemento actual en la lista ordenada
        add_in_order(sorted_list, current['info'], sort_crit)
        current = current['next']  # Pasamos al siguiente nodo

    # Copiamos los elementos de la lista ordenada de vuelta a la lista original
    my_list['first'] = sorted_list['first']
    my_list['last'] = sorted_list['last']
    my_list['size'] = sorted_list['size']

    return my_list

def add_in_order(sorted_list, elem, sort_crit):
    new_node = new_single_node(elem)
    if is_empty(sorted_list) or sort_crit(elem, sorted_list['first']['info']):
        new_node['next'] = sorted_list['first']
        sorted_list['first'] = new_node
        if sorted_list['size'] == 0:  # Si la lista estaba vacía
            sorted_list['last'] = new_node
    else:
        current = sorted_list['first']
        while current['next'] is not None and not sort_crit(elem, current['next']['info']):
            current = current['next']

        new_node['next'] = current['next']
        current['next'] = new_node

        if new_node['next'] is None:  # Si el nuevo nodo es el último
            sorted_list['last'] = new_node

    sorted_list['size'] += 1

def shell_sort(my_list, sort_crit):
    n = my_list['size']
    gap = n // 2  # Se inicia con un hueco de la mitad

    while gap > 0:
        for i in range(gap, n):
            temp = get_element(my_list, i)  # Elemento a insertar
            j = i

            # Mover los elementos que están en la posición de hueco hacia arriba
            while j >= gap and sort_crit(temp, get_element(my_list, j - gap)):
                change_info(my_list, j, get_element(my_list, j - gap))
                j -= gap
            
            change_info(my_list, j, temp)  # Insertar el elemento temporal en su posición
            
        gap //= 2  # Reducir el hueco

    return my_list


def merge_sort(my_list, sort_crit):
    if my_list['size'] < 2:  # Si la lista tiene menos de 2 elementos, ya está ordenada.
        return my_list

    mid = my_list['size'] // 2
    left_half = sub_list(my_list, 0, mid)
    right_half = sub_list(my_list, mid, my_list['size'] - mid)

    left_half = merge_sort(left_half, sort_crit)
    right_half = merge_sort(right_half, sort_crit)

    return merge(left_half, right_half, sort_crit)

def merge(left, right, sort_crit):
    merged = new_list()
    while not is_empty(left) and not is_empty(right):
        if not sort_crit(get_first_element(left), get_first_element(right)):
            add_last(merged, remove_first(left))
        else:
            add_last(merged, remove_first(right))

    while not is_empty(left):  # Agregamos los elementos restantes de la lista izquierda
        add_last(merged, remove_first(left))

    while not is_empty(right):  # Agregamos los elementos restantes de la lista derecha
        add_last(merged, remove_first(right))

    return merged


def quick_sort(my_list, sort_crit):
    if my_list['size'] < 2:  # Si la lista tiene menos de 2 elementos, ya está ordenada.
        return my_list

    return quick_sort_recursive(my_list, 0, my_list['size'] - 1, sort_crit)

def quick_sort_recursive(my_list, lo, hi, sort_crit):
    if lo < hi:
        p = partition(my_list, lo, hi, sort_crit)  # Partición
        quick_sort_recursive(my_list, lo, p - 1, sort_crit)  # Ordenar la parte izquierda
        quick_sort_recursive(my_list, p + 1, hi, sort_crit)  # Ordenar la parte derecha

def partition(my_list, lo, hi, sort_crit):
    pivot = get_element(my_list, hi)  # Elegimos el último elemento como pivote
    i = lo - 1  # Índice del elemento menor

    current = my_list['first']
    for _ in range(lo):
        current = current['next']

    for j in range(lo, hi):
        if sort_crit(current['info'], pivot):
            i += 1
            exchange(my_list, i, j)  # Intercambiamos elementos
        current = current['next']

    exchange(my_list, i + 1, hi)  # Llevamos el pivote a su posición correcta
    return i + 1
"""
  Estructura que contiene la información a guardar en una lista encadenada
"""


def new_single_node(element):
    """ Estructura que contiene la información a guardar en una lista encadenada

        :param element: Elemento a guardar en el nodo
        :type element: any

        :returns: Nodo creado
        :rtype: dict
    """
    node = {'info': element, 'next': None}
    return (node)


#def get_element(node):
    """ Retorna la información de un nodo

        :param node: El nodo a examinar
        :type node: list_node

        :returns: La información almacenada en el nodo
        :rtype: any
    """
    return node['info']


def new_double_node(element):
    """ Estructura que contiene la información a guardar en un nodo de una lista doblemente encadenada
    """
    node = {'info': element,
            'next': None,
            'prev': None
            }
    return node

