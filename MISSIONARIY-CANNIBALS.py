import heapq

def isValid(state):
    M, C, B = state
    if M < 0 or C < 0 or M > 3 or C > 3:
        return False
    if C > M and M > 0:
        return False
    if (3 - C) > (3 - M) and (3 - M) > 0:
        return False
    return True

def successors(state):
    M, C, B = state
    moves = []
    if B == 1:
        if isValid((M, C - 2, 0)):
            moves.append((M, C - 2, 0))
        if isValid((M - 2, C, 0)):
            moves.append((M - 2, C, 0))
        if isValid((M - 1, C - 1, 0)):
            moves.append((M - 1, C - 1, 0))
        if isValid((M, C - 1, 0)):
            moves.append((M, C - 1, 0))
        if isValid((M - 1, C, 0)):
            moves.append((M - 1, C, 0))
    else:
        if isValid((M, C + 2, 1)):
            moves.append((M, C + 2, 1))
        if isValid((M + 2, C, 1)):
            moves.append((M + 2, C, 1))
        if isValid((M + 1, C + 1, 1)):
            moves.append((M + 1, C + 1, 1))
        if isValid((M, C + 1, 1)):
            moves.append((M, C + 1, 1))
        if isValid((M + 1, C, 1)):
            moves.append((M + 1, C, 1))
    return moves

def heuristic(state):
    M, C, B = state
    return (M + C - 2)//2

def astar(start_state):
    heap = []
    heapq.heappush(heap, (heuristic(start_state), 0, [start_state]))
    visited = set()
    while heap:
        _, cost, path = heapq.heappop(heap)
        current_state = path[-1]
        if current_state in visited:
            continue
        if current_state == (0, 0, 0):
            return path
        visited.add(current_state)
        for child in successors(current_state):
            if child not in visited:
                new_path = path + [child]
                new_cost = cost + 1
                heapq.heappush(heap, (new_cost + heuristic(child), new_cost, new_path))
    return None

def print_actions(path):
    for i in range(1, len(path)):
        prev_state = path[i-1]
        current_state = path[i]
        move = (current_state[0] - prev_state[0], current_state[1] - prev_state[1], current_state[2] - prev_state[2])
        if move[2] == 1:
            boat_side = "left"
        else:
            boat_side = "right"
        missionaries = abs(move[0])
        cannibals = abs(move[1])
        if missionaries == 0:
            if cannibals == 1:
                print(f"Move 1 cannibal to the {boat_side} side")
            else:
                print(f"Move {cannibals} cannibals to the {boat_side} side")
        elif cannibals == 0:
            if missionaries == 1:
                print(f"Move 1 missionary to the {boat_side} side")
            else:
                print(f"Move {missionaries} missionaries to the {boat_side} side")
        else:
            print(f"Move {missionaries} missionaries and {cannibals} cannibals to the {boat_side} side")

start_state = 3, 3, 1
solution = astar(start_state)
if solution:
    print("Solution found!")
    print_actions(solution)

    print("Everyone reached to other side of the bank :-) ")
else:
    print("No solution found.")

