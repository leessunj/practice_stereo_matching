class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])

    def add_neighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    def get_connections(self):
        return list(self.adjacent.keys()) #vertex object list

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
        self.distance = {}


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

    def get_num_vertices(self):
        return self.num_vertices

    def get_num_edges(self):
        return len(self.edge_dict)

    def add_edge(self, frm, to, cost = 0):
        if frm not in self.vert_dict:
            self.add_vertex(frm)
        if to not in self.vert_dict:
            self.add_vertex(to)

        self.edge_dict[(frm,to)] = cost
        self.vert_dict[frm].add_neighbor(self.vert_dict[to], cost)
        self.vert_dict[to].add_neighbor(self.vert_dict[frm], cost)
        self.distance[(frm,to)] = 1

    def get_vertices(self):
        return list(self.vert_dict.keys())

    def remove_edge(self,frm,to):
        if frm not in self.vert_dict:
            return
        if to not in self.vert_dict:
            return

        if (frm,to) in self.edge_dict:
            del self.edge_dict[(frm,to)]
            del self.distance[(frm, to)]
        else:
            del self.edge_dict[(to,frm)]
            del self.distance[(to,frm)]

        self.vert_dict[frm].remove_neighbor(self.vert_dict[to])
        self.vert_dict[to].remove_neighbor(self.vert_dict[frm])

    def remove_vertex(self, v):
        if v not in self.vert_dict:
            print("no",v)
            return
        self.num_vertices=self.num_vertices-1
        neighbors=self.vert_dict[v].get_connections()
        for n in neighbors:
            n_id=n.get_id()
            self.remove_edge(n_id,v)
        del self.vert_dict[v]

    def get_edges(self):
        return list(self.edge_dict.keys())

    def get_weight(self,e):
        if e in self.edge_dict:
            return self.edge_dict[e]
        elif (e[1],e[0]) in self.edge_dict:
            return self.edge_dict[(e[1],e[0])]
        else:
            return None

    def weight_sort(self,reverse=False):
        if type(self.edge_dict)!=dict:
            print('not dictionary it is',type(self.edge_dict))
        self.edge_dict=dict(sorted(self.edge_dict.items(), key = lambda item: item[1], reverse = reverse)) #sorted의 결과는 list이다!
        # print(self.edge_dict)

    def get_MST(self):
        self.weight_sort()
        e=self.get_edges()[0]
        print("it's",e,type(e[0]),e[1],self.edge_dict[e])

        MST=Graph()
        MST.add_edge(e[0],e[1],cost=self.edge_dict[e])
        incident={}

        def add_edge_neighbors(nei_dict, e):
            for n in self.get_vertex(e[0]).get_connections():
                if n.get_id()==e[1] or (n.get_id(),e[0]) in nei_dict:
                    continue
                nei_dict[(e[0], n.get_id())] = self.get_weight((e[0], n.get_id()))
            for n in self.get_vertex(e[1]).get_connections():
                if n.get_id()==e[0] or (n.get_id(),e[1]) in nei_dict:
                    continue
                nei_dict[(e[1], n.get_id())] = self.get_weight((e[1], n.get_id()))
            return dict(sorted(nei_dict.items(), key=lambda item: item[1]))

        incident=add_edge_neighbors(incident,e)

        print(f"MST: {MST.get_edges()}edges and {MST.get_num_vertices()} and incident: {incident}")

        n = MST.get_num_vertices()
        while n<self.num_vertices:
            # print(f"incident list {incident}")
            edge=list(incident.keys())[0]

            if (edge[0],edge[1]) in MST.edge_dict or (edge[1],edge[0]) in MST.edge_dict:
                del incident[edge]
                continue
            else:
                MST.add_edge(edge[0],edge[1],cost=incident[edge])
            del incident[edge]
            if n<MST.get_num_vertices():
                incident=add_edge_neighbors(incident,edge)
                # print(f"added {edge}")
            else:#이전보다 vertex가 증가하지 않았다(cycle)
                MST.remove_edge(edge[0],edge[1])
                # print(f"removed {edge}")
                continue
            n = MST.get_num_vertices()
            #print(f"{n}: {edge}=>{MST.get_weight(edge)} *** {MST.get_weight((edge[0],edge[1]))} || type{type(edge)} & {type(MST.get_weight(edge))}") #{MST.get_weight(edge)} == {MST.get_weight((edge[0],edge[1]))
            print(f"{n}:{MST.get_num_edges()} >>{MST.get_vertex(edge[0])}=>{len(MST.get_vertex(edge[0]).get_connections())} | {MST.get_vertex(edge[1])} ")

        return MST

    def get_distance(self, v1, v2):
        if v1==v2:
            return 0
        if (v1, v2) in self.distance:
            return self.distance[(v1, v2)]
        elif (v2, v1) in self.distance:
            return self.distance[(v2, v1)]

        togo = self.get_vertex(v1).get_connections()
        print(f"{v1} neighbors {togo}")
        visited = set()
        dist=[]
        for k in togo:
            vk=k.get_id()
            if v1<vk:
                dist.append(self.get_distance(v1, vk) + self.get_distance(vk, v2))
        if len(dist)<1:
            print(f"NO neighbors between {v1} and {v2}")
            return 999
        self.distance[(v1,v2)]=min(dist)
        return self.distance[(v1,v2)]





import cv2
import math
import networkx as nx

im = cv2.imread('tsukuba/scene1.row3.col1.ppm',0)  # left
im2 = cv2.imread('tsukuba/scene1.row3.col2.ppm',0)  # right
im_graph = Graph()
im2_graph = Graph()
height,width=im.shape # 288 * 384 = 110592
#make image to graph
for i in range(height-1):
    for j in range(width-1):
        im_graph.add_edge((i,j),(i,j+1),cost=abs(int(im[i][j])-int(im[i][j+1]))) #horizontal
        im_graph.add_edge((i,j),(i+1,j),cost=abs(int(im[i][j])-int(im[i+1][j]))) #vertical
for i in range(height-1):
    im_graph.add_edge((i, width-1), (i+1, width-1), cost=abs(int(im[i][-1]) - int(im[i+1][-1])))
for j in range(width-1):
    im_graph.add_edge((height-1,j),(height-1,j+1),cost=abs(int(im[-1][j])-int(im[-1][j+1])))
#len(edges)=2*(height-1)*(width-1)+height+width-2= 220512


#make MST from graph
mst=Graph()
MST=im_graph.get_MST() #실행시간 약 10분쯤
MST.weight_sort()
n=MST.get_num_vertices()
es=MST.get_edges()
print()
print("=====>")
print(f"edges: {len(es)} vertices: {n} max_weight: {es[-1]}=>{MST.get_weight(es[-1])} min_weight: {es[0]}=>{MST.get_weight(es[0])}")
print(f"0,0~2,0: {MST.get_distance((0,0),(2,0))} ")

#find S(p,q)
def similarity(p,q,sigma=1.0):
    global MST
    return math.exp(-MST.get_distance(p,q)/sigma)
#find Cd
def costAggregate(p,d):
    return

def cost(p,d):
    return


#find CAd ~Fri(10/20)
#non-local disparity refinement