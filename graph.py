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

    def get_degree(self):
        return len(self.adjacent)

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
        return list(self.edge_dict.keys())

    def get_weight(self,e):
        if e in self.edge_dict:
            return self.edge_dict[e]
        else:
            return None

    def weight_sort(self):
        if type(self.edge_dict)!=dict:
            print('not dictionary it is',type(self.edge_dict))
        self.edge_dict=dict(sorted(self.edge_dict.items(), key = lambda item: item[1], reverse = True)) #sorted의 결과는 list이다!
        # print(self.edge_dict)

import cv2
import networkx as nx

im = cv2.imread('tsukuba/scene1.row3.col1.ppm',0)  # left
im2 = cv2.imread('tsukuba/scene1.row3.col2.ppm',0)  # right
im_graph = Graph()
im2_graph = Graph()
height,width=im.shape # 288 * 384 = 110592
#make image to graph
for i in range(height-1):
    for j in range(width-1):
        im_graph.add_edge((i,j),(i,j+1),cost=abs(int(im[i][j])-int(im[i][j+1])))
        im_graph.add_edge((i,j),(i+1,j),cost=abs(int(im[i][j])-int(im[i+1][j])))
for i in range(height-1):
    im_graph.add_edge((i, -1), (i+1, -1), cost=abs(int(im[i][-1]) - int(im[i+1][-1])))
for j in range(width-1):
        im_graph.add_edge((-1,j),(-1,j+1),cost=abs(int(im[-1][j])-int(im[-1][j+1])))
#len(edges)=2*(height-1)*(width-1)+height+width-2= 220512

#make MST from graph
im_graph.weight_sort()
edges=im_graph.get_edges()
n=height*width
print(edges[0],im_graph.get_weight(edges[0]))
while len(edges)>n:
    e=edges[0]
    if im_graph.get_vertex(e[0]).get_degree()>1 and im_graph.get_vertex(e[1]).get_degree()>1:
        im_graph.remove_edge(e[0],e[1])
    edges=im_graph.get_edges()
print('after',edges[0],im_graph.get_weight(edges[0]))
#말도 안됨. 36분동안 실행안되는 거는 알고리즘에 문제 있다
