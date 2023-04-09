# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        # 10 points for every food you eat
        """
        Returns a Grid of boolean food indicator variables.

        Grids can be accessed via list notation, so to check
        if there is food at (x,y), just call

        currentFood = state.getFood()
        if currentFood[x][y] == True: ...
        """
        newCapsule = successorGameState.getCapsules()
        # 200 points for every ghost you eat
        # but no point for capsule

        # For Ghost
        newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # Position of ghost do not change regardless of your state
        # because you can't predict the future
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        # Count down from 40 moves
        # ghostStartPos = [ghostState.start.getPosition() for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        evalScore = successorGameState.getScore() # default score

        # Chasing the scaredGhost, avoiding normal Ghost
        #
        # Insight: this implementation helps avoiding ghost only.
        nearGhosts = filter(lambda s: manhattanDistance(s.getPosition(), newPos) <= 1, newGhostStates)
        evalScore += sum(map(lambda s: 200 if s.scaredTimer else -200, nearGhosts))

        # Eat the capsule if possible to catch the ghost, boost actions that moving close to scared ghost.
        for c in newCapsule:
            distance = min(map(lambda g: manhattanDistance(c, g), ghostPositions), default=float('inf'))
            if distance < 20:
                evalScore += 40 / manhattanDistance(c, newPos)

        for g in newGhostStates:
            distance = manhattanDistance(g.getPosition(), newPos)
            if (2 * distance) < g.scaredTimer:
                evalScore += 200

        # Eat the foods, considering nearest food position.
        #
        # Insight 1: cannot set the numerator larger than 10 (advantage when ate a food),
        #            otherwise, pacman will choose STOP rather than MOVE.
        distanceToNearestFood = min(map(lambda c: manhattanDistance(c, newPos), newFood.asList()), default=0)
        evalScore += 10 / distanceToNearestFood if distanceToNearestFood else 0

        return evalScore
        # please change the return score as the score you want

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, gameState: GameState, depth: int, agentIndex: int):
        # Halt situation: when cost for searching is run out or it's games end state.
        if gameState.isWin() or gameState.isLose() or not depth:
            return (None, self.evaluationFunction(gameState))

        # Prepare meta-data for diving in next level.
        actions = gameState.getLegalActions(agentIndex)
        isPacman = (agentIndex == 0)
        numAgents = gameState.getNumAgents()

        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = depth - int(nextAgentIndex == 0)

        # Expand successor nodes
        #
        # Use list to maintain action and state of children node, because the bottleneck
        # of minimax algorithm in here is depth, not width.
        maximizer = (lambda l: max(l, key=lambda x: x[1]))
        minimizer = (lambda l: min(l, key=lambda x: x[1]))
        optimizer = maximizer if isPacman else minimizer

        successors = map(lambda a: gameState.generateSuccessor(agentIndex, a), actions)
        successors = map(lambda s: self.minimax(s, nextDepth, nextAgentIndex)[1], successors)
        successors = zip(actions, successors)
        return optimizer(successors)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        action, _ = self.minimax(gameState, self.depth, self.index)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphabeta(self, gameState: GameState, depth: int, agentIndex: int, alpha: float, beta: float):
        # Halt situation: when cost for searching is run out or it's games end state.
        if gameState.isWin() or gameState.isLose() or not depth:
            return (None, self.evaluationFunction(gameState))

        # Prepare meta-data for diving in next level.
        isPacman = (agentIndex == 0)
        numAgents = gameState.getNumAgents()

        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = depth - int(nextAgentIndex == 0)

        # Expand successor nodes
        #
        # Use list to maintain action and state of children node, because the bottleneck
        # of minimax algorithm in here is depth, not width.
        #
        # TODO: Can this algorithm be implemented by reduce()?
        if isPacman:
            optimal = (None, float('-inf'))
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                _, score = self.alphabeta(successor, nextDepth, nextAgentIndex, alpha, beta)

                optimal = max(optimal, (action, score), key=lambda x: x[1])
                if optimal[1] > beta:
                    return optimal

                alpha = max(alpha, optimal[1])
        else:
            optimal = (None, float('inf'))
            for action in gameState.getLegalActions(agentIndex):
                successor = gameState.generateSuccessor(agentIndex, action)
                _, score = self.alphabeta(successor, nextDepth, nextAgentIndex, alpha, beta)

                optimal = min(optimal, (successor, score), key=lambda x: x[1])
                if optimal[1] < alpha:
                    return optimal

                beta = min(beta, optimal[1])

        return optimal

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        minimal, maximal = float('-inf'), float('inf')
        action, _ = self.alphabeta(gameState, self.depth, self.index, minimal, maximal)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, gameState: GameState, depth: int, agentIndex: int):
        # Halt situation: when cost for searching is run out or it's games end state.
        if gameState.isWin() or gameState.isLose() or not depth:
            return (None, self.evaluationFunction(gameState))

        # Prepare meta-data for diving in next level.
        actions = gameState.getLegalActions(agentIndex)
        isPacman = (agentIndex == 0)
        numAgents = gameState.getNumAgents()

        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = depth - int(nextAgentIndex == 0)

        # Expand successor nodes
        #
        # Use list to maintain action and state of children node, because the bottleneck
        # of minimax algorithm in here is depth, not width.
        maximizer = (lambda l: max(l, key=lambda x: x[1]))
        expect = (lambda l: (None, sum([v for _, v in l]) / len(l)))
        evaluator = maximizer if isPacman else expect

        successors = map(lambda a: gameState.generateSuccessor(agentIndex, a), actions)
        successors = map(lambda s: self.expectimax(s, nextDepth, nextAgentIndex)[1], successors)
        successors = list(zip(actions, successors))
        return evaluator(successors)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, _ = self.expectimax(gameState, self.depth, self.index)
        return action
