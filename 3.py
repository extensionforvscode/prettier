class Game:
    def __init__(self, n):
        temp = []
        self.n = n
        for i in range(n):
            temp2 = []
            for j in range(n):
                temp2.append(0)
            temp.append(temp2)
        self.chessboard = temp
    
    def isSafe(self, state, i, j):
        if self.isColumnSafe(state, i, j) and self.isLeftDiagonalSafe(state, i, j) and self.isRightDiagonalSafe(state, i, j):
            return True
        return False
    
    def isColumnSafe(self, state, i, j):
        while i >= 0:
            if self.chessboard[i][j]:
                return False
            i = i - 1
        return True
    
    def isLeftDiagonalSafe(self, state, i, j):
        i = i - 1
        j = j - 1
        while i >= 0 and j >= 0:
            if self.chessboard[i][j]:
                return False
            i = i - 1
            j = j - 1
        return True
    
    def isRightDiagonalSafe(self, state, i, j):
        i = i - 1
        j = j + 1
        while i >= 0 and j < self.n:
            if self.chessboard[i][j]:
                return False
            i = i - 1
            j = j + 1
        return True
    
    def PlaceQueen(self, r):
        if r == self.n:
            print('Chessboard')
            for i in range(self.n):
                for j in range(self.n):
                    print(self.chessboard[i][j], end=' ')
                print()
            print()
            return True
    
        for c in range(self.n):
            if self.isSafe(self.chessboard, r, c):
                self.chessboard[r][c] = 1
                if self.PlaceQueen(r + 1):
                    return True
                self.chessboard[r][c] = 0
        
        return False

G = Game(5)
G.PlaceQueen(0)
