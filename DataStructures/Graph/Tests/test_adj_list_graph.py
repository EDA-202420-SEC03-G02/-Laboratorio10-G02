import pytest
from DataStructures.Utils.utils import handle_not_implemented
from DataStructures.Graph import adj_list_graph as gl
from DataStructures.Map import map_linear_probing as mp
from DataStructures.List import array_list as lt
from DataStructures.Graph import edge


def setup_tests():
    empty_graph = gl.new_graph()
    some_graph = gl.new_graph()

    ed_1_2 = edge.new_edge(1, 2, 3.0)
    ed_2_1 = edge.new_edge(2, 1, 3.0)

    edges_1 = lt.new_list()
    lt.add_last(edges_1, ed_1_2)
    edges_2 = lt.new_list()
    lt.add_last(edges_2, ed_2_1)

    mp.put(some_graph["vertices"], 1, edges_1)
    mp.put(some_graph["information"], 1, {"name": "A"})
    mp.put(some_graph["vertices"], 2, edges_2)
    mp.put(some_graph["information"], 2, {"name": "B"})

    some_graph["edges"] = 1

    return empty_graph, some_graph



def test_new_graph():
    graph = gl.new_graph(10, False)
    assert graph["edges"] == 0
    assert graph["in_degree"] == None
    assert graph["vertices"]["capacity"] == mp.new_map(10, 0.5)["capacity"]
    assert graph["vertices"]["type"] == "PROBING"
    assert graph["information"]["capacity"] == mp.new_map(10, 0.5)["capacity"]
    assert graph["information"]["type"] == "PROBING"



def test_insert_vertex():
    empty_graph, some_graph = setup_tests()

    gl.insert_vertex(empty_graph, 1, {"name": "A"})

    assert empty_graph["vertices"]["size"] == 1
    assert empty_graph["information"]["size"] == 1

    gl.insert_vertex(some_graph, 3, {"name": "C"})
    assert some_graph["vertices"]["size"] == 3
    assert some_graph["information"]["size"] == 3

    gl.insert_vertex(some_graph, 1, {"name": "D"})
    assert some_graph["vertices"]["size"] == 3
    assert some_graph["information"]["size"] == 3
    pass



def test_num_vertices():
    empty_graph, some_graph = setup_tests()

    assert gl.num_vertices(empty_graph) == 0
    assert gl.num_vertices(some_graph) == 2



def test_num_edges():
    empty_graph, some_graph = setup_tests()

    assert gl.num_edges(empty_graph) == 0

    assert gl.num_edges(some_graph) == 1



def test_vertices():
    empty_graph, some_graph = setup_tests()

    vertices = gl.vertices(empty_graph)

    assert lt.size(vertices) == 0

    vertices = gl.vertices(some_graph)

    assert lt.size(vertices) == 2
    assert vertices["elements"] is not None



def test_edges():
    empty_graph, some_graph = setup_tests()

    edges = gl.edges(empty_graph)

    assert lt.size(edges) == 0

    edges = gl.edges(some_graph)

    assert lt.size(edges) == 1
    assert edges["elements"] is not None



def test_degree():
    empty_graph, some_graph = setup_tests()

    assert gl.degree(empty_graph, 1) is None

    assert gl.degree(some_graph, 0) is None
    assert gl.degree(some_graph, 1) == 1
    assert gl.degree(some_graph, 2) == 1



def test_in_degree():
    empty_graph, some_graph = setup_tests()

    assert gl.in_degree(empty_graph, 1) is None

    assert gl.in_degree(some_graph, 0) is None
    assert gl.in_degree(some_graph, 1) == 1
    assert gl.in_degree(some_graph, 2) == 1



def test_add_edge():
    empty_graph, some_graph = setup_tests()

    gl.add_edge(empty_graph, 1, 2, 3.0)

    assert gl.num_edges(empty_graph) == 0

    gl.insert_vertex(some_graph, 3, {"name": "D"})
    gl.add_edge(some_graph, 1, 3, 3.0)

    assert gl.num_edges(some_graph) == 2
