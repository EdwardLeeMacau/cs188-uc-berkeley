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
import math

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
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

    # def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here. (question 1)

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here. (question 1)

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        """
        Comment:
        The evaluation depends on currentState and the action
          > Pacman can't know what the ghost will do.
          > The successorState only return the result after Pacman move
          - When Pacman move next to ghosts, be careful!
        """
        foods = currentGameState.getFood().asList()

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()

        newGhostStates = successorGameState.getGhostStates()

        evalScore = float(0)

        # Chasing the scaredGhost
        # If calculate the distance, the agent will have a bug shaking.. (or standing)
        for ghostState in newGhostStates:
            ghostPosition = ghostState.getPosition()
            distance = manhattanDistance(ghostPosition, newPos)

            if distance <= 1:
                if ghostState.scaredTimer != 0:
                    evalScore += 200
                else:
                    evalScore -= 200

        # Eat the capsule
        for capsule in currentGameState.getCapsules():
            distance = manhattanDistance(capsule, newPos)
            if distance == 0:
                evalScore += 100
            else:
                evalScore += 50 / distance

        # Eat the foods, considering next food position.
        for foodPosition in foods:
            distance = manhattanDistance(foodPosition, newPos)

            if distance == 0:
                evalScore += 100
            else:
                evalScore += 1 / (distance ** 2)

        """
        # Sometimes pacman fall-in stop action
        # Bug shaking....
        if pos != newPos:
            evalScore += 1
        """

        return evalScore

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

def negation(alist):
    return map(lambda x: -x, alist)

def judge(scoreDepths, agentIndex):
    """ Received the list contains tuples: [(score, Depths), (score, Depths)...].
        To evaluate which one is better.

        Use in Minimax / alpha-beta: Choose 1 action by the successor's evaluation.
        Assume all of the agent do the best:
        - Pacman: Win / Keep alive.
        - Ghost:  Lose / Getting Low score.
    """
    isPacman = (agentIndex == 0)

    scores = [scoreDepth[0] for scoreDepth in scoreDepths]
    depths = [scoreDepth[1] for scoreDepth in scoreDepths]

    # Ghost strategy: minimize the score, maximize the death depths
    # Return above result so that Pacman know how to move 'carefully'
    if not (isPacman):
        scores = negation(scores)
        depths = negation(scores)

    # If not all the Indices directed to "Lose", the bestIndices
    # will exclude those indices.
    bestScore = max(scores)
    bestIndices = [index for index in range(0, len(scoreDepths)) if scores[index] == bestScore]

    # Pacman: find a way to stay alive(when it must die) -> min(depth) /
    #         find a way to win quickly or keep alive -> max(depth)
    if bestScore == float("-inf"):
        bestDepth = min([depths[indice] for indice in bestIndices])
    else:
        bestDepth = max([depths[indice] for indice in bestIndices])

    bestIndices = [index for index in range(0, len(scoreDepths)) if (scores[index] == bestScore and depths[index] == bestDepth)]
    chosenIndex = random.choice(bestIndices)
    return chosenIndex

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

    def isPacman(self):
        """ If the agent is pacman, return True, else return false. """
        return self.index == 0

    def minimax(self, gameState, depth, agentIndex):
        """ Given the state and the character's playing stragety,
            Return the score at that node

            In minimaxClassic, the map is small and the agent
            guess that it will die very soon(all the successor
            return -inf).

            Return the tuple (score, depth), the deeper(smaller)
            depth is better.
        """
        isPacman    = (agentIndex == 0)
        isEnd       = (depth == 0)
        numAgents   = gameState.getNumAgents()
        legalMoves  = gameState.getLegalActions(agentIndex)

        # Halt situration: depht enough / Win / Lose
        # if gameState.isWin() or gameState.isLose() or (depth == 1 and isLastGhost):
        if gameState.isWin() or gameState.isLose() or isEnd:
            return self.evaluationFunction(gameState)


        # if gameState.isWin():
        #     return (float("inf"), depth)
        # if gameState.isLose():
        #     return (float("-inf"), depth)
        # if (depth == 1 and isLastGhost):
        #     score = self.evaluationFunction(gameState)
        #     return (score, depth)

        # If don't halt, recursive call minimax function.
        successorStates = map(lambda move: gameState.generateSuccessor(agentIndex, move), legalMoves)

        # Next agent ready to make a decision
        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = (depth - 1) if (nextAgentIndex < agentIndex) else depth

        # Choose 1 action (-> chosenIndex) to maximize(Pacman) / minimize(Ghost) the score.

        # Method 1:
        # scoreDepths = [self.minimax(successorState, nextDepth, nextAgentIndex) for successorState in successorStates]
        # chosenIndex = judge(scoreDepths, agentIndex)
        # return scoreDepths[chosenIndex]\

        # Method 2: (Implement)
        scores = [self.minimax(successorState, nextDepth, nextAgentIndex) for successorState in successorStates]
        bestScore = max(scores) if isPacman else min(scores)
        bestIndices = [index for index in range(0, len(scores)) if scores[index]==bestScore]
        chosenIndex = random.choice(bestIndices)
        return scores[chosenIndex]

    def getAction(self, gameState):
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
        """
        # Search for legalMoves, generaterd the successors.
        legalMoves = gameState.getLegalActions(self.index)
        successorStates = map(lambda move: gameState.generateSuccessor(self.index, move), legalMoves)

        # Find the successor's score by strategy.
        nextAgentIndex = (self.index + 1) % gameState.getNumAgents()
        nextDepth = (self.depth - 1) if (nextAgentIndex < self.index) else self.depth

        # Choose 1 action (-> chosenIndex) to maximize(Pacman) / minimize(Ghost) the score.

        # Method 1:
        # scoreDepths = [self.minimax(successorState, nextDepth, nextAgentIndex) for successorState in successorStates]
        # chosenIndex = judge(scoreDepths, 0)
        # return scoreDepths[chosenIndex]

        # Method 2: (Implement)
        scores = [self.minimax(successorState, nextDepth, nextAgentIndex) for successorState in successorStates]
        bestScore = max(scores)
        bestIndices = [index for index in range(0, len(scores)) if scores[index]==bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def isPacman(self):
        """ If the agent is pacman, return True, else return false. """
        return self.index == 0

    def alphabeta(self, gameState, depth, agentIndex, alpha, beta):
        # Given the state and the character's playing stragety, Return the score
        isPacman = (agentIndex == 0)
        isEnd = (depth == 0)
        numAgents   = gameState.getNumAgents()
        legalMoves  = gameState.getLegalActions(agentIndex)

        # Halt situration: depht enough / Win / Lose
        if gameState.isWin() or gameState.isLose() or isEnd:
            return self.evaluationFunction(gameState)

        # if gameState.isWin():
        #     return (float("inf"), depth)
        # if gameState.isLose():
        #     return (float("-inf"), depth)
        # if (depth == 1 and isLastGhost):
        #     score = self.evaluationFunction(gameState)
        #     return (score, depth)

        # if it's not the last step, recursive call alphabeta function.
        # Next agent ready to make a decision
        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = (depth - 1) if (nextAgentIndex < agentIndex) else depth

        # Alpha-Beta algorithm
        if isPacman:
            # Method 1
            # scoreDepth = (float("-inf"), depth)
            # for move in legalMoves:
                # successor = gameState.generateSuccessor(agentIndex, move)
                # successorScoreDepth = self.alphabeta(successor, nextDepth, nextAgentIndex, alpha, beta)
                # chosenIndex = judge((scoreDepth, successorScoreDepth), agentIndex)
                # scoreDepth = (scoreDepth, successorScoreDepth)[chosenIndex]
                # if (scoreDepth[0] >= beta):
                #     return scoreDepth

                # alpha = max(alpha, scoreDepth[0])
            # return score

            # Method 2
            score = float("-inf")

            for move in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, move)
                successorScore = self.alphabeta(successor, nextDepth, nextAgentIndex, alpha, beta)
                score = max(score, successorScore)
                # if (score >= beta)
                if (score > beta):
                    return score
                alpha = max(alpha, score)

            return score
        else:
            # Method 1
            # scoreDepth = (float("inf"), depth)
            # for move in legalMoves:
                # Successor = gameState.generateSuccessor(agentIndex, move)
                # successorScoreDepth = self.alphabeta(successor, nextDepth, nextAgentIndex, alpha, beta)
                # chosenIndex = judge((scoreDepth, successorScoreDepth), agentIndex)
                # scoreDepth = (scoreDepth, successorScoreDepth)[chosenIndex]
                # if (scoreDepth[0] <= alpha):
                #     return scoreDepth
                # beta = min(beta, scoreDepth[0])

            # return scoreDepth

            # Method 2
            score = float("inf")

            # Search for each successor
            for move in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, move)
                successorScore = self.alphabeta(successor, nextDepth, nextAgentIndex, alpha, beta)
                score = min(score, successorScore)
                # if (score <= alpha)
                if (score < alpha):
                    return score
                beta = min(beta, score)

            return score

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Search for legalMoves, generaterd the successors.
        legalMoves = gameState.getLegalActions(self.index)
        # successorStates = map(lambda move: gameState.generateSuccessor(self.index, move), legalMoves)

        # Find the successor's score by strategy.
        nextAgentIndex = (self.index + 1) % gameState.getNumAgents()
        nextDepth = (self.depth - 1) if (nextAgentIndex < self.index) else self.depth

        # Method 1
        # scoreDepths = map(lambda successorState: self.alphabeta(successorState, nextDepth, nextAgentIndex, float("-inf"), float("inf")), successorStates)
        # chosenIndex = judge(scoreDepths, 0)

        # Method 2
        scores = []
        alpha  = float("-inf")
        beta   = float("inf")
        bestScore = float("-inf")

        for move in legalMoves:
            successorState = gameState.generateSuccessor(self.index, move)
            successorScore = self.alphabeta(successorState, nextDepth, nextAgentIndex, alpha, beta)
            bestScore = max(bestScore, successorScore)

            # Save the successorScore, if some successor return same value, we can randomly choose it.
            scores.append(successorScore)

            # Notes: if (score >= beta): return score
            # But this is impossible in the first node because initilized beta = float("inf").

            # Adjust the lower boundary if it's necessary to search other nodes.
            alpha = max(alpha, bestScore)

        bestIndices = [index for index in range(len(scores)) if scores[index]==bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

def average(scoreDepths):
    # scoreDepths = [(score, depth), (score, depth)...]
    averageScore = 0
    averageDepth = 0
    return (averageScore, averageDepth)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def isPacman(self):
        """ If the agent is pacman, return True, else return false. """
        return self.index == 0

    def average(self, alist):
        return sum(alist) / float(len(alist))

    def expectiMinimax(self, gameState, depth, agentIndex):
        """ Given the state and the character's playing stragety, Return the score
            Pacman: Choose the action optimized the score
            Ghost : Random choosing action
        """
        isPacman    = (agentIndex == 0)
        isEnd       = (depth == 0)
        numAgents   = gameState.getNumAgents()
        legalMoves  = gameState.getLegalActions(agentIndex)

        # Halt situration: depht enough / Win / Lose
        """
        if gameState.isWin():
            return float("inf")
        if gameState.isLose():
            return float("-inf")
        if (depth == 1 and isLastGhost):
            evaluate = self.evaluationFunction(gameState)
            return evaluate
        """

        if gameState.isWin() or gameState.isLose() or isEnd:
            return self.evaluationFunction(gameState)

        # If don't halt, recursive call minimax function.
        successorStates = map(lambda move: gameState.generateSuccessor(agentIndex, move), legalMoves)

        # Next agent ready to make a decision
        nextAgentIndex = (agentIndex + 1) % numAgents
        nextDepth = (depth - 1) if (nextAgentIndex < agentIndex) else depth

        # Method 1:
        # scoreDepths = [self.expectiMinimax(successorState, nextDepth, nextAgentIndex) for successorState in successorStates]
        # averageScoreDepth = judge(scoreDepths, agentIndex)
        # return averageScoreDepth

        # Method 2: (implement)
        scores = [self.expectiMinimax(successorState, nextDepth, nextAgentIndex) for successorState in successorStates]

        if isPacman:
            return max(scores)
        else:
            return sum(scores) / len(scores)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        # Search for legalMoves, generaterd the successors.
        legalMoves = gameState.getLegalActions(self.index)
        successorStates = map(lambda move: gameState.generateSuccessor(self.index, move), legalMoves)

        # Find the successor's score by strategy.
        nextAgentIndex = (self.index + 1) % gameState.getNumAgents()
        nextDepth = (self.depth - 1) if (nextAgentIndex < self.index) else self.depth
        scores = map(lambda successorState: self.expectiMinimax(successorState, nextDepth, nextAgentIndex), successorStates)

        # Choose the best action
        bestScore = max(scores)

        if math.isnan(bestScore):
            bestIndices = [index for index in range(len(scores)) if math.isnan(scores[index])]
        else:
            bestIndices = [index for index in range(len(scores)) if scores[index]==bestScore]

        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
    """
    foods = currentGameState.getFood().asList()
    pacmanPos   = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()

    """
    Basically, I use the stateScore as the reference value.
    This is reasonable, the score represent some siturations, such as:
    1. Eating the foods near the Pacman
    2. Escaping from active pacman (Lose: -500, very negative expectation)
    3. Eat capsule when the ghost is come near
    4. Don't stop too long.

    Another reason, appling the score is a way to represent that eating foods,
    scaredGhost are good action which these actions increase the score.
    (Not as Q1, no parameter 'action' and I can't know whether the foods are eaten)
    """
    evalScore = currentGameState.getScore()

    """
    Remember that eating capsules will not getting any score, but given a
    potential to gain a large score(eat ghost or escape from lose). In here
    I would encourge Pacman to eat the capsules.

    The weight should be higher... Because for each time eat a food get 10 point.
    I have tried 20 and 40, seems 40 gives a better expectation
    """

    # Eat the capsule
    capsulePos = currentGameState.getCapsules()
    if len(capsulePos) > 0:
        capsuleDistance = min([manhattanDistance(capsule, pacmanPos) for capsule in capsulePos])
        evalScore += 40.0 / capsuleDistance

    """
    After eaten capsules, it's able to eat the ghost
    A concern of chasing far ghost is, wasting time an the scardTimer = 0 ...

    Therefore, set a easily evaluation to guess whether the Pacman can(or can't)
    eat the ghost. (Define as canEat)
    """
    # Chasing the scaredGhost
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        distance = manhattanDistance(ghostPos, pacmanPos)

        canEat = int(ghostState.scaredTimer) / 2 > distance

        if canEat:
            if (distance <= 1):
                evalScore += 200
            elif (distance == 2):
                evalScore += 100
            elif (distance == 3):
                evalScore += 50
            elif (distance == 4):
                evalScore == 25

    """
    Sometimes Pacman have eaten all the foods around, and it can't see the
    further foods then stand here, encourage it to go!
    """
    # Eat the foods, considering next food position.
    for foodPosition in foods:
        distance = manhattanDistance(foodPosition, pacmanPos)
        evalScore += 1.0 / distance

    """
    Something done not good is, Pacman don't know it's going into a dead end
    (be flanked). It needs to search and remember the entrance of the lane.
    """
    return evalScore


# Abbreviation
better = betterEvaluationFunction

