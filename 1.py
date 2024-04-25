class Node:
    def __init__(self, name, neighbours):
        self.name = name
        self.neighbours = neighbours
    def getNeighbours(self):
        return self.neighbours
    def getName(self):
        return self.name
def BFS(startNode, goalNode):
    q = [startNode]
    visited = set() 
    print("Final path is ")
    while q:
        currentNode = q.pop(0)
        if currentNode.getName() != goalNode.getName():
            print(f"{currentNode.name} ->", end=" ")  
        else:
            print(f"{currentNode.name}\nNode found !!")
            return 
        visited.add(currentNode)
        n = currentNode.getNeighbours()
        for i in n:
            if i not in visited:
                q.append(i)
    if not q:
        print("\nNode not found !!")
H = Node('H',[])
A = Node('A', [Node('B', []),Node('C', [H]),Node('D', []),Node('E',[]),Node('F',[])])
BFS(A, H)


class Node:
    def __init__(self, name, neighbours):
        self.name = name
        self.neighbours = neighbours

    def getName(self):
        return self.name

    def getNeighbours(self):
        return self.neighbours

def DLS(currentNode, goalNode, depth, max_depth, openList=[]):
    openList.append(currentNode.getName())
    if currentNode.getName() == goalNode.getName():
        return True
    if depth >= max_depth:
        return False
    for neighbour in currentNode.getNeighbours():
        result = DLS(neighbour, goalNode, depth + 1, max_depth, openList)
        if result:
            return True

    return False

node1 = Node("A", [])
node2 = Node("B", [node1])
node3 = Node("C", [node2])
node4 = Node("D", [node3])

openList = []
found = DLS(node4, node1, 0, 3, openList)
print(found)  
print(openList) 