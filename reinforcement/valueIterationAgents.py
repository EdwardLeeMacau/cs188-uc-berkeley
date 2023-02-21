# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
            Your value iteration agent should take an mdp on
            construction, run the indicated number of iterations
            and then act according to the resulting policy.

            Some useful mdp methods you will use:
                mdp.getStates() Ex: ['TERMINAL_STATE', (0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
                mdp.getPossibleActions(state) Ex: ('north', 'west', 'south', 'east')
                mdp.getTransitionStatesAndProbs(state, action) Given the state and action, get the probabilities of next possible states, Ex: [((1, 2), 0.2), ((2, 2), 0.8)]
                mdp.getReward(state, action, nextState) Get the reward of the state
                mdp.isTerminal(state) Check if the state is terminal
        """
        self.mdp = mdp
        self.discount = discount        # discount rate
        self.iterations = iterations
        self.values = util.Counter()    # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for state in mdp.getStates(): 
            self.values[state] = 0

        for index in range(0, iterations):
            newValues = util.Counter()

            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state): continue

                possibleActions = self.mdp.getPossibleActions(state)
                newValues[state] = max(map(lambda action: self.getQValue(state, action), possibleActions))
                
            self.values = newValues
                
    def getValue(self, state): #utility
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        Q = []
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        
        for stateAndProbs in transitionStatesAndProbs:
            nextState = stateAndProbs[0]
            nextProb  = stateAndProbs[1]
            Q.append(nextProb * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState]))

        # Finally compute the summation of the Probs * (Rewards + discout * utiltiy)
        Q = sum(Q)
        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if state == "TERMINAL_STATE": return
        
        possibleActions   = self.mdp.getPossibleActions(state)
        actionsExpectation = util.Counter()

        for action in possibleActions:
            transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
            expectation = []

            for nextState, prob in transitionStatesAndProbs:
                reward = self.mdp.getReward(state, action, nextState)
                expectation.append(prob * (reward + self.values[nextState]))
            
            actionsExpectation[action] = sum(expectation)

        bestAction = actionsExpectation.argMax()
        return bestAction
        
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
