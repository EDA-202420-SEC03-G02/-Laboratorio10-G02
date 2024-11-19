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
    Calcula un hash para una llave utilizando el método MAD:
    hash_value(y) = ((a * y + b) % p) % M.
    
    Parámetros:
    - table (dict): Tabla hash con los parámetros necesarios.
    - key (any): Llave a la que se le calculará el hash.
    
    Retorna:
    - int: Valor del hash.
    """
    # Parámetros necesarios para el método MAD
    scale = table['scale']
    shift = table['shift']
    prime = table['prime']
    size = table['size']
    
    # Cálculo del hash
    h = hash(key)
    return ((scale * h + shift) % prime) % size

    return h
import random

def initialize_table(size):
    """
    Inicializa una tabla hash con los parámetros necesarios para el método MAD.
    
    :param size: Tamaño de la tabla hash.
    :type size: int
    :return: Tabla hash inicializada.
    :rtype: dict
    """
    prime = next_prime(size)  # Encuentra el siguiente número primo mayor a 'size'
    return {
        "scale": random.randint(1, prime - 1),  # a > 0
        "shift": random.randint(0, prime - 1),  # b
        "prime": prime,
        "capacity": size,  # M
        **{i: [] for i in range(size)}  # Inicializa las celdas vacías
    }

def next_prime(n):
    """
    Encuentra el siguiente número primo mayor o igual a n.
    """
    def is_prime(num):
        if num < 2:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    while not is_prime(n):
        n += 1
    return n
