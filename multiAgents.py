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

        An unimplemented ReflexAgent is driven by getting all the legal moves from the
        current position, getting a potential future score from each of those moves,
        (which move will vary with different score by food, no food, presence of ghost, etc.)
        and return the score for each moves. getAction() simply picks the move with the highest
        score. It does not consider any future impacts from choosing the action, just the
        score from taking the immediate action.

        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        # unimplemented version simply gets the score not considering the ghost, food,
        # scared ghost, etc.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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

        Without a valid evaluation function, I noticed that when pacman is not
        surrounded by food, it has trouble finding a path and goes back and forth
        since every action gives the same "best" score, and it picks it randomly.
        Giving the illusion that its moving back and forth with no sense of route.

        .getFood():
        Returns a Grid of boolean food indicator variables.

        Grids can be accessed via list notation, so to check
        if there is food at (x,y), just call

        currentFood = state.getFood()
        if currentFood[x][y] == True: ...

        the action parameter is the successor move from the current state

        This function gives the score from taking the "action" move from
        the currrent state by evaluating the current game env
        """

        score = 100

        # successorGameState: Give the state as a grid from taking the action
        # use successorGameState.getScore() to get the immediate score
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newSuccessorPosition = successorGameState.getPacmanPosition()
        pacmanPosition = currentGameState.getPacmanPosition()

        # Pacman does not know what to do if there are no food pallets adjacent to him
        # Return .getScore() if food is directly adjacent him
        # If not, give Manhattan distance to the closest food pallet

        capsules = currentGameState.getCapsules()

        hasFood = currentGameState.getFood()
        x = pacmanPosition[0]
        y = pacmanPosition[1]
        if newSuccessorPosition in capsules:
            return 10000
        elif hasFood[x + 1][y]:
            return score + successorGameState.getScore()

        elif hasFood[x][y + 1]:
            return score + successorGameState.getScore()

        elif hasFood[x - 1][y]:
            return score + successorGameState.getScore()
        elif hasFood[x][y - 1]:
            return score + successorGameState.getScore()
        else:
            # No more adjacent food! Find the closest food
            closestFood = (float('inf'), float('inf'))
            minDist = float('inf')
            for x in range(24):
                for y in range(len(hasFood[x])):
                    if hasFood[x][y]:
                        manDist = manhattanDistance(pacmanPosition, (x, y))
                        if manDist < minDist:
                            minDist = manDist
                            closestFood = (x, y)

            # smaller the distance the better
            # reward if taking the new position got us closer to the closest food
            # penalize if opposite
            prevDistance = manhattanDistance(pacmanPosition, closestFood)
            distance = manhattanDistance(newSuccessorPosition, closestFood)

            if prevDistance < distance:
                score = score - 100
            elif distance < prevDistance:
                score = score + 100
            else:
                score = score - 10

            # decrement if action is Stop
            if action == "Stop":
                score = score - 100

            # Pacman should move towards food pallet but according to the distance between the ghost
            ghostPosition = successorGameState.getGhostPositions()[0]

            # I want to motivate Pacman to eat more food, not be scared about ghosts
            # so only penalize a move if it's way too close to the ghost
            distGhost = manhattanDistance(ghostPosition, newSuccessorPosition)
            if distGhost < 2:
                score = score - 500

            return score

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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)

    Score the terminal state (tree leaves) with self.evaluationFunction.
    Use self.depth and self.evaluationFunction to generate the miniMax tree
    and the values in the terminal states.
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

        Base cases:
        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


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
