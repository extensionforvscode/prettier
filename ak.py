#!/usr/bin/env python
# coding: utf-8

# # N Queens

# In[21]:


def is_safe(row, col, board, n):
  r, c = row, col
  while r >= 0:
    if board[r][c] == 1:
      return False
    r = r - 1
    
  r, c = row, col
  while r >= 0 and c >= 0:
    if board[r][c] == 1:
      return False
    r = r - 1
    c = c - 1
    
  r, c = row, col
  while r >= 0 and c < n:
    if board[r][c] == 1:
      return False
    r = r - 1
    c = c + 1
    
  return True

def chess_board(r, n, board):
  if r >= n:
    print_solution(board)
    return False
  
  for c in range(n):
    if is_safe(r, c, board, n):
      board[r][c] = 1
      chess_board(r+1, n, board)
      board[r][c] = 0
      
  return False

def print_solution(board):
  global solutions
  solutions += 1
  print(f"Solution {solutions}: ")
  for i in range(len(board)):
    for j in range(len(board)):
      if board[i][j] == 1:
        print("Q", end=" ")
      else:
        print(".", end=" ")
    print()
  print()
  
if __name__ == "__main__":
  n = int(input("Enter length of chess board or number of queens :- "))
  board = [[0 for _ in range(n)] for _ in range(n)]
  solutions = 0
  
  if not chess_board(0, n, board):
    print("No solution is found")


# # MiniMax Algorithm

# In[18]:


def minimax(depth, nodeIndex, values, maximizing, alpha, beta):
  if depth == 3:
    return values[nodeIndex]
  
  if maximizing:
    best = -1000
    for i in range(2):
      val = minimax(depth+1, nodeIndex*2 + i,values,  False, alpha, beta)
      best = max(best, val)
      alpha = max(alpha, best)
      
      if beta <= alpha:
        break
    return best
  else:
    best = 1000
    for i in range(2):
      val = minimax(depth+1, nodeIndex*2 + i,values,  True, alpha, beta)
      best = min(best, val)
      beta = min(beta, best)
      
      if beta <= alpha:
        break
    return best
  
if __name__ == "__main__":
  values = [3, 5, 6, 9, 1, 2, 0, -1]
  val = minimax(0, 0, values, True, -1000, 1000)
  print(f"Optimal value is : {val}")


# # BFS

# In[32]:


from collections import deque
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

def BFS(start):
  queue = []
  visited = set()
  queue.append(start)
  visited.add(start)
  
  while queue:
    node = queue.pop(0)
    print(node, end="->")
    for neighbor in graph[node]:
      if neighbor not in visited:
        queue.append(neighbor)
        visited.add(neighbor)

BFS('A')    


# # DLS

# In[4]:


graph = {
    'A': {'B': 10, 'C': 15},
    'B': {'D': 12},
    'C': {'E': 10},
    'D': {'F': 5},
    'E': {'G': 7},
    'F': {},
    'G': {}
}

def DLS(graph, start, target, depth, depth_limit):
    print(start, end=" ")
    if start == target:
        return True
    if depth >= depth_limit:
        return False
    for neighbour, _ in graph[start].items():
        if DLS(graph, neighbour, target, depth+1, depth_limit):
            return True
    return False

DLS(graph, "A", "G", 0, 3)


# # Best First Search

# In[97]:


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

def Best_First_Search(graph, start, goal, heuristic):
  visited = set()
  heap = [(0, start)]
  path = []
  while heap:
    cost, node = heapq.heappop(heap)
    path.append(node)
    if node == goal:
      return path
    visited.add(node)

    for neighbour, neighbour_cost in graph[node].items():
      if neighbour not in visited:
        heapq.heappush(heap, (heuristic[neighbour], neighbour))

path = Best_First_Search(graph, "A", "G", heuristics)
print(path)


# # A Star

# In[89]:


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

def Astar(start, target):
  visited = set()
  open_list = [(0, start, [start])]
  
  while open_list:
    cost, node, path = heapq.heappop(open_list)
    
    if node == target:
      return cost, path
    
    visited.add(node)
    for neighbour, neighbour_cost in graph[node].items():
      if neighbour not in visited:
        heuristic_cost = heuristics[neighbour]
        new_cost = cost + neighbour_cost
        total_new_cost = new_cost + heuristic_cost
        new_path = path + [neighbour]
        heapq.heappush(open_list, (total_new_cost, neighbour, new_path))
        
print(Astar("A", "G"))


# # 8 Puzzle

# In[90]:


start_state = [[1, 2, 3], [0, 5, 6], [4, 7, 8]]
goal_state = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
def heuristic(state):
  c = 0
  for i in range(3):
    for j in range(3):
      if goal_state[i][j] != state[i][j]:
        c += 1
  return c

def find_empty_tile(state):
  for i in range(3):
    for j in range(3):
      if state[i][j] == 0:
        return i, j
      
def find_possible_moves(state):
  possible_solutions = []
  i, j = find_empty_tile(state)
  if 3>i+1>=0 and 3>j>=0:
    possible_solutions.append(swap_tiles(state, i, j, i+1, j))
  if 3>i-1>=0 and 3>j>=0:
    possible_solutions.append(swap_tiles(state, i, j, i-1, j))
  if 3>i>=0 and 3>j+1>=0:
    possible_solutions.append(swap_tiles(state, i, j, i, j+1))
  if 3>i>=0 and 3>j-1>=0:
    possible_solutions.append(swap_tiles(state, i, j, i, j-1))
  return possible_solutions
    
def swap_tiles(state, row1, col1, row2, col2):
  new_state = [row.copy() for row in state]
  new_state[row1][col1], new_state[row2][col2] = new_state[row2][col2], new_state[row1][col1]
  return new_state

def solve(start_state, goal_state):
  visited = set()
  open_list = [(heuristic(start_state), start_state, [start_state])]
  
  while open_list:
    current_cost, current_state, path = heapq.heappop(open_list)
    
    if current_state == goal_state:
      return path
    visited.add(tuple(map(tuple, current_state)))
    
    for move in find_possible_moves(current_state):
      if tuple(map(tuple, move)) not in visited:
        newcost = cost + heuristic(move)
        newpath = path + [move]
        heapq.heappush(open_list, (newcost, move, newpath))
        
sol = solve(start_state, goal_state)
sol


# # Missionary

# In[1]:


import heapq
def is_valid(state):
  m, c, b = state
  if m<0 or m>3 or c<0 or c>3:
    return False
  if c>m and m>0:
    return False
  if(3-c)>(3-m) and (3-m)>0:
    return False
  return True

def moves(state):
  m, c, b = state
  moves = []
  if b==0:
    if is_valid((m+2, c, 1)):
      moves.append((m+2, c, 1))
    if is_valid((m, c+2, 1)):
      moves.append((m, c+2, 1))
    if is_valid((m+1, c+1, 1)):
      moves.append((m+1, c+1, 1))
    if is_valid((m+1, c, 1)):
      moves.append((m+1, c, 1))
    if is_valid((m, c+1, 1)):
      moves.append((m, c+1, 1))
  else:
    if is_valid((m-2, c, 0)):
      moves.append((m-2, c, 0))
    if is_valid((m, c-2, 0)):
      moves.append((m, c-2, 0))
    if is_valid((m-1, c-1, 0)):
      moves.append((m-1, c-1, 0))
    if is_valid((m-1, c, 0)):
      moves.append((m-1, c, 0))
    if is_valid((m, c-1, 0)):
      moves.append((m, c-1, 0))
  return moves

def heuristic(state):
  m, c, b = state
  return (m+c-2)//2

def solve(start_state):
  visited = set()
  open_list = [(heuristic(start_state), 0, [start_state])]
  
  while open_list:
    # print(str(open_list[-1][2]))
    _, cost, path = heapq.heappop(open_list)
    current_state = path[-1]
    
    if current_state == (0, 0, 0):
      return cost, path
    
    visited.add(current_state)
    
    for move in moves(current_state):
      if move not in visited:
        new_cost = cost + 1 + heuristic(move)
        new_path = path + [move]
        heapq.heappush(open_list, (heuristic(move), new_cost, new_path))
        
path = solve((3, 3, 1))
path


# # Bayesian Network

# In[13]:


from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

cpd_e = TabularCPD(variable="e", variable_card=2, values=[[0.6], [0.4]])
cpd_i = TabularCPD(variable="i", variable_card=2, values=[[0.7], [0.3]])
cpd_m = TabularCPD(variable="m", variable_card=2, evidence=["e", "i"], evidence_card=[2, 2], values=[[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]])
cpd_s = TabularCPD(variable="s", variable_card=2, evidence=["i"], evidence_card=[2], values=[[0.95, 0.2], [0.05, 0.8]])
cpd_a = TabularCPD(variable="a", evidence=["m"], evidence_card=[2], variable_card=2, values=[[0.8, 0.1], [0.2, 0.9]])

model = BayesianNetwork([("e", "m"), ("i", "m"), ("i", "s"), ("m", "a")])
model.add_cpds(cpd_e, cpd_i, cpd_m, cpd_s, cpd_a)

print(f"Model is Consistent :- {model.check_model()}")

for cpd in model.get_cpds():
  print(f"CPS for {cpd.variable} : \n{cpd}",)
  
  
from pgmpy.inference import VariableElimination

infer = VariableElimination(model)

print("Probability of admission given marks 1 :- ")
print(infer.query(variables=["a"], evidence={"m":1}).values[1])

print("Probability of marks given exam 0 but iq 1 :- ")
print(infer.query(variables=["m"], evidence={"e":0, "i":1}).values[1])


# # Hill Climbing

# In[20]:


import copy

def heuristic(state, goal):
  goal_ = goal[3]
  h = 0 
  for stack in state:
    for i in range(len(stack)):
      if goal_[i] == stack[i]:
        h += i
      else:
        h -= i
        
  return h

visited =[]
def next_move(current, goal, prev_heuristic):
  global visited
  state = copy.deepcopy(current)
  for i in range(len(state)):
    temp = copy.deepcopy(state)
    if len(temp[i]) > 0:
      elem = temp[i].pop()
      for j in range(len(temp)):
        temp1 = copy.deepcopy(temp)
        if j != i:
          temp1[j] = temp1[j] + [elem]
          if temp1 not in visited:
            current_heuristic = heuristic(temp1, goal)
            if current_heuristic > prev_heuristic:
              child = copy.deepcopy(temp1)
              return child
  return 0

def solve(init_state, goal_state):
  global visited
  if(init_state == goal_state):
    print(goal_state)
    return
  
  current =  copy.deepcopy(init_state)
  while True:
    visited.append(copy.deepcopy(current))
    print(current)
    prev = heuristic(current, goal_state)
    child = next_move(current, goal_state, prev)
    if child==0:
      print(current)
      return
    current  = copy.deepcopy(
      child)
    

global visited_states
init_state = [['C'],['D'],[],['A','B']]
goal_state = [[],[],[],['A','B','C','D']]
solve(init_state, goal_state)


# In[ ]:




