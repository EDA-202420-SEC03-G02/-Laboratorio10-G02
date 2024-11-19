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