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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        # apply the evaluation function on each legalMove and choose the action with the best score.
        # action variable a parameter for legalMoves to be passed to the function
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # save the index with the best score
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        # If there are multiple moves with the same best score, randomly pick

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        # In order to improve the reflex agent, I need to improve the evaluation function
        # using the food location / ghost location
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def maximizer(state, agentIndex, depth):
            value = -float('inf')
            chosenAction = None
            possibleActions = state.getLegalActions(agentIndex)

            # If I am at the last agent (opponent) next agent should be
            # my minimax agent
            nextAgent = agentIndex + 1
            if state.getNumAgents() - 1 == agentIndex:
                nextAgent = 0

            for action in possibleActions:
                nextState = state.generateSuccessor(agentIndex, action)
                result = miniMax(nextState, nextAgent, depth + 1, action)
                if value < result[0]:
                    value = result[0]
                    # Replace the return value: chosenAction that was chosen
                    chosenAction = action
            return [value, chosenAction]

        def minimizer(state, agentIndex, depth):
            value = float('inf')
            chosenAction = None

            nextAgent = agentIndex + 1
            if state.getNumAgents() - 1 == agentIndex:
                nextAgent = 0

            possibleActions = state.getLegalActions(agentIndex)
            for action in possibleActions:
                nextState = state.generateSuccessor(agentIndex, action)
                result = miniMax(nextState, nextAgent, depth + 1, action)
                if value > result[0]:
                    value = result[0]
                    chosenAction = action
            return [value, chosenAction]

        def miniMax(state, agentIndex, currentDepth, action):

            # base case 1: Reached depth
            # Few ways to conclude full depth. One way is # of agent * self.depth
            # Keep track of the depth every recursive call
            if state.getNumAgents() * self.depth <= currentDepth:
                # Return as a list of value and action that was previously passed
                # This seems unnecessary as we only need the action for the root (maximizer)
                # at layer 1. I can't think of a more efficient method right now
                # Stick it with passing it every recursive return
                return [self.evaluationFunction(state), action]

            # base case 2: Terminal state
            if state.isWin() or state.isLose():
                return [self.evaluationFunction(state), action]
            if agentIndex == 0:
                return maximizer(state, agentIndex, currentDepth)
            elif agentIndex > 0:
                return minimizer(state, agentIndex, currentDepth)

        # return: [value, chosenAction]
        return miniMax(gameState, 0, 0, None)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
