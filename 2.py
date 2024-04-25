class Node:
    def __init__(self, name, heuristic, neighbours):
        self.name = name
        self.heuristic = heuristic
        self.neighbours = neighbours

    def getName(self):
        return self.name

    def getNeighbours(self):
        return self.neighbours

    def getHeuristic(self):
        return self.heuristic

def BFS(startNode, goalNode):
    q = [startNode]
    visited = set()
    closed_list = []
    while q:
        min_cost = float('inf')
        min_node = None
        for node in q:
            if node.getHeuristic() < min_cost:
                min_node = node
                min_cost = node.getHeuristic()
        if min_node.getName() == goalNode.getName():
            print(closed_list)
            print("Node Found")
            return 
        q.remove(min_node)
        closed_list.append(min_node.getName())
        visited.add(min_node)
        for neighbour in min_node.getNeighbours():
            if neighbour not in visited and neighbour not in q:
                q.append(neighbour)
    print("Node not found")

node_A = Node('A', 8, [])
node_B = Node('B', 6, [])
node_C = Node('C', 5, [])
node_D = Node('D', 3, [])
node_E = Node('E', 4, [])
node_F = Node('F', 2, [])
node_G = Node('G', 1, [])
node_H = Node('H', 1, [])
node_A.neighbours = [node_B, node_C]
node_B.neighbours = [node_D, node_E]
node_C.neighbours = [node_F, node_G]
BFS(node_A, node_F)


import heapq
graph = {
    'A': {'B': 10, 'C': 15},
    'B': {'D': 12},
    'C': {'E': 10},
    'D': {'F': 5},
    'E': {'G': 7},
    'F': {},
    'G': {}
}
heuristics = {'A': 78, 'B': 40, 'C': 30, 'D': 25, 'E': 20, 'F': 10, 'G': 0}
start = 'A'
goal = 'G'
def Astar(start, goal):
    visited = set()
    openList = [(0, start, [start])]  
    while openList:
        total_cost, node, path = heapq.heappop(openList)
        if node == goal:
            return total_cost
        if node not in visited:
            visited.add(node)
            for neighbor, cost in graph[node].items():
                if neighbor not in visited:
                    heuristic_cost = heuristics[neighbor]
                    new_cost = total_cost + cost
                    total_new_cost = new_cost + heuristic_cost
                    new_path = path + [neighbor]
                    heapq.heappush(openList, (total_new_cost, neighbor, new_path))
print(Astar(start, goal))
