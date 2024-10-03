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
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newCapsules = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        inf = float('inf')
        if successorGameState.isWin():
            return inf
        if successorGameState.isLose():
            return -inf

        # if this action resulted eating a food, bonus points
        if successorGameState.getNumFood() < currentGameState.getNumFood():
            score += 300

        # closer the food, greater score
        min_food = inf
        for food in newFood.asList():
            min_food = min(min_food, manhattanDistance(food, newPos))
        score += (1 / min_food)

        newGhostPositions = successorGameState.getGhostPositions()
        min_ghost = inf
        for ghost_pos in newGhostPositions:
            distance = manhattanDistance(newPos, ghost_pos)
            # if distance to ghost is close, minus points
            if distance <= 1:
                score -= 500
            min_ghost = min(min_ghost, distance)
        # further the closest ghost, better score
        score -= (1 / min_ghost)
        score += sum(newScaredTimes)
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
    """

    # minimax implementation (dispatch)
    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self.minimax_action = None

    def max_value(self, gameState, agent, depth):
        # initialize v = -∞
        v = float('-inf')
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            minimax_value = self.minimax(successor, agent + 1, depth)
            v = max(v, minimax_value)
            # if this is the best minimax action save it, so pacman can perform it at depth 1
            if depth == 1 and v == minimax_value:
                self.minimax_action = action
        return v

    def min_value(self, gameState, agent, depth):
        # initialize v = ∞
        v = float('inf')
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            v = min(v, self.minimax(successor, agent + 1, depth))
        return v

    def minimax(self, gameState, agent, depth):
        # if the state is a terminal state: return the state’s utility
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # with every min/max call agentIndex is incremented by 1.
        # to find real index of agent we should mod()
        agent = agent % gameState.getNumAgents()
        if agent == 0:
            # agent is pacman, should find max from next depth
            if depth < self.depth:
                return self.max_value(gameState, agent, depth + 1)
            else:
                # search depth is achieved, should not expand further
                return self.evaluationFunction(gameState)
        else:
            # agent is ghost, should find min, we are on the same depth
            # (multiple min layers (one for each ghost) for every max layer)
            return self.min_value(gameState, agent, depth)

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
        self.minimax(gameState, 0, 0)
        return self.minimax_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self.alpha_beta_action = None

    def max_value(self, gameState, agent, depth, alpha, beta):
        v = float('-inf')
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            alpha_beta_minimax_value = self.alpha_beta_minimax(successor, agent + 1, depth, alpha, beta)
            v = max(v, alpha_beta_minimax_value)
            if depth == 1 and v == alpha_beta_minimax_value:
                self.alpha_beta_action = action
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, gameState, agent, depth, alpha, beta):
        v = float('inf')
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            v = min(v, self.alpha_beta_minimax(successor, agent + 1, depth, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
        return v

    def alpha_beta_minimax(self, gameState, agent, depth, alpha, beta):
        # if the state is a terminal state: return the state’s utility
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # with every min/max call agentIndex is incremented by 1.
        # to find real index of agent we should mod()
        agent = agent % gameState.getNumAgents()
        if agent == 0:
            # agent is pacman, should find max from next depth
            if depth < self.depth:
                return self.max_value(gameState, agent, depth + 1, alpha, beta)
            else:
                # search depth is achieved, should not expand further
                return self.evaluationFunction(gameState)
        else:
            # agent is ghost, should find min, we are on the same depth
            # (multiple min layers (one for each ghost) for every max layer)
            return self.min_value(gameState, agent, depth, alpha, beta)

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # alpha is max so should start with -inf
        # beta is min so should start with inf
        self.alpha_beta_minimax(gameState, 0, 0, float("-inf"), float("inf"))
        return self.alpha_beta_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        super().__init__(evalFn, depth)
        self.expectimax_action = None

    def max_value(self, gameState, agent, depth):
        # initialize v = -∞
        v = float('-inf')
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            expectimax_value = self.expectimax(successor, agent + 1, depth)
            v = max(v, expectimax_value)
            # if this is the best minimax action save it, so pacman can perform it at depth 1
            if depth == 1 and v == expectimax_value:
                self.expectimax_action = action
        return v

    def probability(self, legalActions):
        return 1 / len(legalActions)

    def exp_value(self, gameState, agent, depth):
        v = 0
        legalActions = gameState.getLegalActions(agent)
        for action in gameState.getLegalActions(agent):
            successor = gameState.generateSuccessor(agent, action)
            p = self.probability(legalActions)
            v += p * self.expectimax(successor, agent + 1, depth)
        return v

    def expectimax(self, gameState, agent, depth):
        # if the state is a terminal state: return the state’s utility
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        # with every min/max call agentIndex is incremented by 1.
        # to find real index of agent we should mod()
        agent = agent % gameState.getNumAgents()
        if agent == 0:
            # agent is pacman, should find max from next depth
            if depth < self.depth:
                return self.max_value(gameState, agent, depth + 1)
            else:
                # search depth is achieved, should not expand further
                return self.evaluationFunction(gameState)
        else:
            # agent is ghost, should find min, we are on the same depth
            # (multiple min layers (one for each ghost) for every max layer)
            return self.exp_value(gameState, agent, depth)

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.expectimax(gameState, 0, 0)
        return self.expectimax_action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: modified evaluation function from reflex agent
    - is terminal state
        if state is win infinite reward, if lose -infinity
    - number and location of food
        closer the food pellet, better score
        smaller amount of food left, better score
    - number and location of capsules (similar to food)
    - position of ghosts
        if ghost is too close(manhattan distance smaller than 1, score -500) and not scared minus points
    - scared time


    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    inf = float('inf')
    if currentGameState.isWin():
        return inf
    if currentGameState.isLose():
        return -inf

    # closer the food, greater score
    min_food = inf
    for food in food:
        min_food = min(min_food, manhattanDistance(food, pos))
    score += (1 / min_food)
    score -= currentGameState.getNumFood()

    min_capsule = inf
    for capsule in capsules:
        min_capsule = min(min_capsule, manhattanDistance(capsule, pos))
    score += (1 / min_capsule)
    score -= len(capsules)

    newGhostPositions = currentGameState.getGhostPositions()
    for ghost_pos in newGhostPositions:
        distance = manhattanDistance(pos, ghost_pos)
        if distance <= 1 and sum(scaredTimes) <= 1:
            score -= 500
    score += sum(scaredTimes)

    return score


# Abbreviation
better = betterEvaluationFunction
