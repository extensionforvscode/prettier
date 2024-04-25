import copy

visited_states = []
def heuristic(curr_state,goal_state):
    goal_=goal_state[3]
    val = 0
    for state in curr_state:
        for i in range(len(state)):
            if goal_[i]==state[i]:
                val+=i
            else:
                val-=i
    return val
            
def generate_next(curr_state,prev_heu,goal_state):
    global visited_states
    state = copy.deepcopy(curr_state)
    for i in range(len(state)):
        temp = copy.deepcopy(state)
        if len(temp[i]) > 0:
            elem = temp[i].pop()
            for j in range(len(temp)):
                temp1 = copy.deepcopy(temp)
                if j != i:
                    temp1[j] = temp1[j] + [elem]
                    if (temp1 not in visited_states):
                        curr_heu=heuristic(temp1,goal_state)
                        if curr_heu>prev_heu:
                            child = copy.deepcopy(temp1)
                            return child
    return 0

def solution_(init_state,goal_state):
    global visited_states
    if (init_state == goal_state):
        print (goal_state)
        print("solution found!")
        return
    
    current_state = copy.deepcopy(init_state)
    while(True):
        visited_states.append(copy.deepcopy(current_state))
        print(current_state)
        prev_heu=heuristic(current_state,goal_state)
        child = generate_next(current_state,prev_heu,goal_state)
        if child==0:
            print("Final state - ",current_state)
            return
        current_state = copy.deepcopy(child)  

def solver():
    global visited_states
    init_state = [['C'],['D'],[],['A','B']]
    goal_state = [[],[],[],['A','B','C','D']]
    solution_(init_state,goal_state)

solver()