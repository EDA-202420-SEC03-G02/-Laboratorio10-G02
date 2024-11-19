#Linear Probing 425 libro
import random



def new_map(num_elements, load_factor, prime=109345121):
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
    """Ingresa una pareja llave-valor en la tabla de hash con sondeo lineal."""
    index = hash_value(my_map, key)  # Calcula el índice usando hash_value
    table = my_map['table']
    capacity = my_map['capacity']
    
    # Crear la pareja llave-valor
    pareja = (key, value)
    
    # Sondeo lineal para encontrar una posición vacía o la misma clave
    initial_index = index
    while table[index] is not None:
        stored_key, _ = table[index]
        
        # Si la llave ya existe, reemplaza el valor
        if stored_key == key:
            table[index] = pareja
            return my_map
        
        # Sondeo lineal: mover al siguiente índice
        index = (index + 1) % capacity
        
        # Verificación para evitar un ciclo infinito
        if index == initial_index:
            raise Exception("Hash table is full")

    # Si encontramos una posición vacía, insertamos la nueva pareja llave-valor
    table[index] = pareja
    my_map['size'] += 1
    my_map['current_factor'] = my_map['size'] / capacity

    # Verificar si es necesario rehash (cuando el factor de carga se excede)
    if my_map['current_factor'] > my_map['limit_factor']:
        rehash(my_map)  # Completar rehash si el factor de carga se excede
    return my_map


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
    M = table["size"]

    h = (hash(key) * a + b) % p  # Método MAD
    return h % M


def new_map_entry(key, value):
    """ Retorna una pareja llave valor para ser guardada en un Map

        :param key: Llave de la pareja
        :type key: any
        :param value: Valor de la pareja
        :type value: any

        :return: Una entrada con la pareja llave-valor
        :rtype: map_entry
    """
    entry = {'key': key, 'value': value}
    return entry


def set_key(my_entry, key):
    """ Asigna un valor nuevo a la ``key`` del entry recibido como parámetro

        :param my_entry: La pareja llave-valor
        :type my_entry: map_entry
        :param key: La nueva llave
        :type key: any

        :return: La pareja modificada
        :rtype: map_entry
    """
    my_entry['key'] = key
    return my_entry


def set_value(my_entry, value):
    """Asigna un valor nuevo al ``value`` del entry recibido como parámetro

        :param my_entry: La pareja llave-valor
        :type my_entry: map_entry
        :param value: El nuevo value
        :type value: any

        :return: La pareja modificada
        :rtype: map_entry
    """
    my_entry['value'] = value
    return my_entry


def get_key(my_entry):
    """ 
    Retorna la llave de la entry recibida como parámetro

    :param my_entry: La pareja llave-valor
    :type my_entry: map_entry

    :return: La llave de la pareja
    :rtype: any
    """
    return my_entry['key']


def get_value(my_entry):
    """
    Retorna el valor de la entry recibida como parámetro

    :param my_entry: La pareja llave-valor
    :type my_entry: map_entry
    
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
