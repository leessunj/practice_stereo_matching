class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return self.adjacent.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.adjacent[neighbor]

    def remove_neighbor(self, neighbor):
        del self.adjacent[neighbor]

class Graph:
    def __init__(self):
        self.vert_dict = {}
        self.edge_dict = {}
        self.num_vertices = 0


    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vert_dict[node] = new_vertex
        return new_vertex

    def get_vertex(self, n):
        if n in self.vert_dict:
            return self.vert_dict[n]
        else:
            return None

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.edge_dict[(frm,to)] = cost
        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)

    def get_vertices(self):
        return self.vert_dict.keys()

    def remove_edge(self,frm,to):
        if frm not in self.vert_dict:
            return
        if to not in self.vert_dict:
            return

        if (frm,to) in self.edge_dict:
            del self.edge_dict[(frm,to)]
        else:
            del self.edge_dict[(to,frm)]

        self.vert_dict[frm].remove_neighbor(self.vert_dict[to])
        self.vert_dict[to].remove_neighbor(self.vert_dict[frm])

    def remove_vertex(self, v):
        if v not in self.vert_dict:
            print("no",v)
            return
        self.num_vertices=self.num_vertices-1
        neighbors=list(self.vert_dict[v].get_connections())
        for n in neighbors:
            n_id=n.get_id()
            self.remove_edge(n_id,v)
        del self.vert_dict[v]

    def get_edges(self):
        return self.edge_dict.keys()

    def get_weight(self,e):
        if e in self.edge_dict:
            return self.edge_dict[e]
        else:
            return None

    def weight_sort(self):
        self.edge_dict=sorted(self.edge_dict, key = lambda item: item[1], reverse = True)

import cv2

im = cv2.imread('tsukuba/scene1.row3.col1.ppm',0)  # left
im2 = cv2.imread('tsukuba/scene1.row3.col2.ppm',0)  # right
im_graph = Graph()
im2_graph = Graph()
height,width=im.shape

for i in range(height-1):
    for j in range(width-1):
        im_graph.add_edge((i,j),(i,j+1),cost=abs(int(im[i][j])-int(im[i][j+1])))
        im_graph.add_edge((i,j),(i+1,j),cost=abs(int(im[i][j])-int(im[i+1][j])))
im_graph.weight_sort()
#왜 add할 때까지만 해도 잘 있던 딕셔너리가 사라지는지 모르겠음;;왜 리스트가 되지?? 어디서부터 잘못된 것임??
#edges=im_graph.get_edges()
#print(edges[:3],type(edges),im_graph.get_weight(edges[0]),im_graph.get_weight(edges[1]))



# #make MST
# for element in im_graph
#
# g.add_vertex('a')
# g.add_vertex('b')
# g.add_vertex('c')
# g.add_vertex('d')
# g.add_vertex('e')
# g.add_vertex('f')
#
# g.add_edge('a', 'b', 7)
# g.add_edge('a', 'c', 9)
# g.add_edge('a', 'f', 14)
# g.add_edge('b', 'c', 10)
# g.add_edge('b', 'd', 15)
# g.add_edge('c', 'd', 11)
# g.add_edge('c', 'f', 2)
# g.add_edge('d', 'e', 6)
# g.add_edge('e', 'f', 9)
# g.remove_vertex('a')
# g.remove_vertex('a')
# for v in g:
#     for w in v.get_connections():
#         vid = v.get_id()
#         wid = w.get_id()
#         print('( %s , %s, %3d)'  % ( vid, wid, v.get_weight(w)))
#
# for v in g:
#     print( 'g.vert_dict[%s]=%s' %(v.get_id(), g.vert_dict[v.get_id()]))