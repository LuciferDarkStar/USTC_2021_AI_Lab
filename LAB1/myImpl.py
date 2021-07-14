import util

"""
Data sturctures we will use are stack, queue and priority queue.

Stack: first in last out
Queue: first in first out
    collection.push(element): insert element
    element = collection.pop() get and remove element from collection

Priority queue:
    pq.update('eat', 2)
    pq.update('study', 1)
    pq.update('sleep', 3)
pq.pop() will return 'study' because it has highest priority 1.

"""

"""
problem is a object has 3 methods related to search state:

problem.getStartState()
Returns the start state for the search problem.

problem.isGoalState(state)
Returns True if and only if the state is a valid goal state.

problem.getChildren(state)
For a given state, this should return a list of tuples, (next_state,
step_cost), where 'next_state' is a child to the current state, 
and 'step_cost' is the incremental cost of expanding to that child.

"""
def myDepthFirstSearch(problem):
    visited = {}
    frontier = util.Stack()

    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]                
        
        if state not in visited:
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))

    return []

def myBreadthFirstSearch(problem):
    # YOUR CODE HERE
    visited = {}
    frontier = util.Queue()

    frontier.push((problem.getStartState(), None))

    while not frontier.isEmpty():
        state, prev_state = frontier.pop()

        if problem.isGoalState(state):
            solution = [state]
            while prev_state != None:
                solution.append(prev_state)
                prev_state = visited[prev_state]
            return solution[::-1]                
        
        if state not in visited:
            visited[state] = prev_state

            for next_state, step_cost in problem.getChildren(state):
                frontier.push((next_state, state))
    #util.raiseNotDefined()
    return []

def myAStarSearch(problem, heuristic):
    # YOUR CODE HERE
    frontier = util.PriorityQueue()
    start = [problem.getStartState(), heuristic(problem.getStartState()), []]
    p = 0
    frontier.push(start, p)  # queue push at index_0
    closed = []
    while not frontier.isEmpty():
        [state, cost, path] = frontier.pop()
        # print(state)
        if problem.isGoalState(state):
            # print(path)
            return path+[state]  # here is a deep first algorithm in a sense
        if state not in closed:
            closed.append(state)
            for child_state, child_cost in problem.getChildren(state):
                new_cost = cost + child_cost
                new_path = path + [state]
                frontier.push([child_state, new_cost, new_path], new_cost + heuristic(child_state))
    #util.raiseNotDefined()
    return []

"""
Game state has 4 methods we can use.

state.isTerminated()
Return True if the state is terminated. We should not continue to search if the state is terminated.

state.isMe()
Return True if it's time for the desired agent to take action. We should check this function to determine whether an agent should maximum or minimum the score.

state.getChildren()
Returns a list of legal state after an agent takes an action.

state.evaluateScore()
Return the score of the state. We should maximum the score for the desired agent.

"""
class MyMinimaxAgent():

    def __init__(self, depth):
        self.depth = depth

    def minimax(self, state, depth):
        if depth==0 or state.isTerminated():
            return None, state.evaluateScore()        

        best_state, best_score = None, -float('inf') if state.isMe() else float('inf')

        def Max_s(a,b,c,d):
            if(a>c):
                return a,b
            else:
                return c,d

        def Min_s(a,b,c,d):
            if(a<c):
                return a,b
            else:
                return c,d

        for child in state.getChildren():
            # YOUR CODE HERE
            #util.raiseNotDefined()
            if state.isMe():
                ghost,min_score=self.minimax(child,depth)
                best_score,best_state=Max_s(best_score,best_state,min_score,child)
            elif child.isMe():
                agent,max_score=self.minimax(child,depth-1)
                best_score,best_state=Min_s(best_score,best_state,max_score,child)
            else:
                ghost,min_score=self.minimax(child,depth)
                best_score,best_state=Min_s(best_score,best_state,min_score,child)
        return best_state, best_score

    def getNextState(self, state):
        best_state, _ = self.minimax(state, self.depth)
        return best_state

class MyAlphaBetaAgent():

    def __init__(self, depth):
        self.depth = depth
    
    def minimax(self, state, depth,a,b):
        if depth==0 or state.isTerminated():
            return None, state.evaluateScore()        

        best_state, best_score = None, -float('inf') if state.isMe() else float('inf')

        def Max_s(a,b,c,d):
            if(a>c):
                return a,b
            else:
                return c,d

        def Min_s(a,b,c,d):
            if(a<c):
                return a,b
            else:
                return c,d

        for child in state.getChildren():
            # YOUR CODE HERE
            #util.raiseNotDefined()
            if state.isMe():
                ghost,min_score=self.minimax(child,depth,a,b)
                best_score,best_state=Max_s(best_score,best_state,min_score,child)
                if best_score > b:
                    return best_state, best_score
                a = max(a, best_score)
            elif child.isMe():
                agent,max_score=self.minimax(child,depth-1,a,b)
                best_score,best_state=Min_s(best_score,best_state,max_score,child)
                if best_score < a:
                    return best_state, best_score
                b = min(b, best_score)
            else:
                ghost,min_score=self.minimax(child,depth,a,b)
                best_score,best_state=Min_s(best_score,best_state,min_score,child)
                if best_score < a:
                    return best_state, best_score
                b = min(b, best_score)
        return best_state, best_score

    def getNextState(self, state):
        # YOUR CODE HERE
        #util.raiseNotDefined()
        best_state, _ = self.minimax(state, self.depth,-float('inf'), float('inf'))
        return best_state
