import heapq

class puzzle:
    def __init__(self, start, goal):
        self.start = start
        self.goal = goal

    def heuristics(self, state):
        penalty = 0
        for i in range(0, 3):
            for j in range(0, 3):
                if state[i][j] != self.goal[i][j] and state[i][j] != '_':
                    penalty += 1
        return penalty

    def findBlank(self, state):
        for i in range(0, 3):
            for j in range(0, 3):
                if state[i][j] == '_':
                    return i, j

    def findPossibleStates(self, state):
        possible_moves = []
        x, y = self.findBlank(state)
        if 3 > x + 1 >= 0 and 3 > y  >= 0:
            possible_moves.append([x + 1, y ])
        if 3 > x - 1 >= 0 and 3 > y >= 0:
            possible_moves.append([x - 1, y ])
        if 3 > x >= 0 and 3 > y - 1 >= 0:
            possible_moves.append([x, y - 1])
        if 3 > x >= 0 and 3 > y + 1 >= 0:
            possible_moves.append([x, y + 1])
        return possible_moves

    def shuffleBlankSpace(self, state, x, y, X, Y):
        temp_state = [row[:] for row in state]  # Create a copy of the state
        temp = temp_state[x][y]
        temp_state[x][y] = temp_state[X][Y]
        temp_state[X][Y] = temp
        return temp_state

    def generateMoves(self, state):
        finalMoves = []
        x, y = self.findBlank(state)
        moves = self.findPossibleStates(state)
        for move in moves:
            X, Y = move
            finalMoves.append(self.shuffleBlankSpace(state, x, y, X, Y))
        return finalMoves

    def solve(self):
        openList = [(self.heuristics(self.start), self.start, [self.start])]
        visited = set()
        while openList:
            currentCost, currentState, path = heapq.heappop(openList)
            visited.add(tuple(map(tuple, currentState)))  # Update visited

            if currentState == self.goal:
                return path

            for move in self.generateMoves(currentState):
                if tuple(map(tuple, move)) not in visited:  # Check if move not visited
                    moveCost = currentCost + self.heuristics(move)
                    heapq.heappush(openList, (moveCost, move, path + [move]))

# Example instantiation and solving
start_state = [['1', '2', '3'], ['_', '5', '6'], ['4', '7', '8']]
goal_state = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '_']]

p = puzzle(start_state, goal_state)
solution = p.solve()
solution