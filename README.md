### Chapter 2: Intelligent Agents

#### 2.1 Introduction

**Definition of an agent**: Anything that can be viewed as perceiving its environment through sensors and acting upon that environment through actuators.

- **Agent function**: f:P∗→Af: P^* \rightarrow Af:P∗→A
    - **P**: Set of all percepts
    - **A**: Set of all actions

**Agent program**: Runs on the physical architecture to implement the agent function.

#### 2.2 Good Behavior: The Concept of Rationality

- **Rational agent**: An agent that does the right thing, which is expected to maximize its performance measure based on the percept sequence it has seen so far.
    - **Performance measure**: A criterion for success of an agent's behavior.
    - **Rationality**: Depends on four factors:
        1. The performance measure defining the criterion of success.
        2. The agent's prior knowledge of the environment.
        3. The actions that the agent can perform.
        4. The agent's percept sequence to date.
#### 2.3 The Nature of Environments

**Properties of environments**:

1. **Fully observable vs. partially observable**:
    
    - **Fully observable**: Sensors provide complete information about the environment's state.
    - **Partially observable**: Sensors provide incomplete information about the environment's state.
2. **Deterministic vs. stochastic**:
    
    - **Deterministic**: The next state of the environment is completely determined by the current state and the agent’s action.
    - **Stochastic**: The next state is not completely determined by the current state and the agent’s action.
3. **Episodic vs. sequential**:
    
    - **Episodic**: The agent's experience is divided into independent episodes.
    - **Sequential**: The agent's current action may affect future actions.
4. **Static vs. dynamic**:
    
    - **Static**: The environment does not change while the agent is deliberating.
    - **Dynamic**: The environment can change while the agent is deliberating.
5. **Discrete vs. continuous**:
    
    - **Discrete**: A limited number of distinct states and actions.
    - **Continuous**: A range of possible states and actions.
6. **Single agent vs. multi-agent**:
    
    - **Single agent**: Only one agent operating in the environment.
    - **Multi-agent**: Multiple agents operating and possibly interacting in the environment.
#### 2.4 The Structure of Agents

- **Simple reflex agents**:
    - Select actions based on the current percept, ignoring the rest of the percept history.
    - **Example**: A vacuum cleaner that turns left if it bumps into an obstacle.
- **Model-based reflex agents**:
    - Maintain an internal state that depends on the percept history.
    - **Example**: A vacuum cleaner that remembers where it has already cleaned.
- **Goal-based agents**:
    - Take actions to achieve goals. Requires planning and search.
    - **Example**: A robot that navigates to a specific location.
- **Utility-based agents**:
    - Use a utility function to evaluate the desirability of different states, allowing for decision-making in complex environments.
    - **Example**: An autonomous car that balances safety, speed, and comfort.


### Chapter 3: Search

#### 3.1 Problem-Solving Agents

- **Problem-solving agent**: Uses search to find a sequence of actions leading to a desirable goal.
- **Problem formulation** involves defining:
    - **Initial state**: Where the agent starts.
    - **Actions**: Set of actions the agent can take.
    - **Goal test**: Determines if the current state is a goal state.
    - **Path cost**: Numeric cost assigned to each path.
	
#### 3.3 Searching for Solutions

- **Search algorithms**: Explore the state space to find a solution path.
- **Tree search**: Explores paths systematically from the initial state to the goal state, keeping track of expanded nodes.
- **Graph search**: Adds the ability to avoid revisiting states by maintaining a list of visited nodes, which helps in preventing cycles and reducing redundant work.

	we measure a search algorithm on 4 things:
	1) Completeness: Is it guaranteed to find a solution
	2) Optimality: Does it find the optimal solution
	3) Time complexity: time
	4) Space complexity: memory used
#### 3.4 Uninformed Search Strategies

**Breadth-first search (BFS)**
- **Process**:
    - Explores all nodes at the present depth level before moving to nodes at the next depth level.
    - Uses a queue (FIFO) to keep track of nodes.
- **Properties**:
    - **Complete**: will find goal node given node is at some depth d and branching factor b is finite
    - **Optimal**: not optimal, shallowest goal is not always the optimal one. Only optimal when all actions have the same cost or if path cost is nondecreasing.
    - **Time complexity**: $O(b^d)$. This would be $O(b^{d+1})$ if we were done when we expanded the goal node.
    - **Space complexity**: $O(b^d)$. Every expanded node is stored in the explored set and every node generated remains in memory leaving $O(b^{b-1})$ nodes in the explored set and $O(b^d)$ in the frontier. 
    - NOTE: switching to tree search would not save much space.

**Uniform-cost search**
- **Process**:
    - Expands node with the lowest path cost g(n).
    - uses priority queue to store frontier ordered by g
    - goal test is applied when it is selected for expansion rather then when generated.
    - test is added in case a better path is found to a node currently on the frontier (replaces more expensive paths)
- **Properties**:
    - **Complete**: complete given that every node has a non zero positive cost
    - NOTE: would get stuck in infinite loop if there is a path with an infinite sequence of zero cost actions
    - **Optimal**: guaranteed to find optimal path to goal node.
    - NOTE: note characterized by depth but by cost
    - **Time complexity**: Let C be the cost of the optimal solution, and c be some small positive constant, then $O(b^{1+\floor{C/c}})$ which could be greater than $O(b^d)$. UCS would explore large tress of small steps before exploring paths with large steps and useful steps. When steps are equal, UCS and BFS are similar except BFS stops at the goal whereas UCS will examine all the nodes at the goal's depth to see if one has a lower cost, thus UCS is strictly more work.
	
**Depth-first search**
- **Process**:
    - Expands deepest node in the current frontier. Once a node has no successors it gets dropped and backs up to a node with unexplored successors
    - Uses a queue (LIFO or stack) to keep track of nodes.
- **Properties**:
    - NOTE: properties are heavily dependent on whether we are looking at a graph or tree
    - **Complete**: 
	    - **Graph**: complete since it will eventually expand every node
	    - **Tree**: not complete. can get stuck in loops (which graph's avoid)
			NOTE: DFS tree search can be modified at no extra memory cost to check new states against those on the path from the root to the current node, avoiding infinite loops but doesn't avoid redundant paths
		NOTE: both would fail in infinite state spaces
    - **Optimal**: both nonoptimal
    - **Time complexity**: $O(b^m)$ where m is the max depth of any node.
- **DFS vs BFS**:
	- For graph search, there is no advantage, but DFS tree search needs to only store a single path from the root to a leaf, along with remaining unexpanded sibling nodes for each node on the path. Once a node is expanded it is removed from memory as soon as all its descendants have been fully explored meaning it takes $O(bm)$ storage.
	
**Iterative deepening depth-first search**
- **Process**:
    - combines BFS with DFS by incrementally increasing the depth and running DFS
    - This modification gets the memory benefits of of DFS O(bd) and the benefits of BFS of being complete
- **Properties**:
    - NOTE: properties are heavily dependent on whether we are looking at a graph or tree
    - **Complete**: 
	    - **Graph**: complete since it will eventually expand every node
	    - **Tree**: not complete. can get stuck in loops (which graph's avoid)
			NOTE: DFS tree search can be modified at no extra memory cost to check new states against those on the path from the root to the current node, avoiding infinite loops but doesn't avoid redundant paths
		NOTE: both would fail in infinite state spaces
    - **Optimal**: both nonoptimal
    - **Time complexity**: $O(b^m)$ where m is the max depth of any node.
- **DFS vs BFS**:
	- For graph search, there is no advantage, but DFS tree search needs to only store a single path from the root to a leaf, along with remaining unexpanded sibling nodes for each node on the path. Once a node is expanded it is removed from memory as soon as all its descendants have been fully explored meaning it takes $O(bm)$ storage.
	- It may seem that this approach is costly, but that is not the case. This is because in tree search with the same or nearly the same branching factor, the majority of nodes are in the bottom level, which given a **time complexity** of $O(b^d)$$ the same as BFS (asymptotically)

**Comparison of uninformed search strategies**
	![[Pasted image 20240726022355.png]]

#### 3.5 Informed (heuristic) search

similar to UCS, the general approach called **best-first search** uses a **evaluation function f(n)** (as compared to g(n)) then evaluates the next lowest cost node first. The choice of f determines the search strategy. Most best-first algorithms include as a component of f a **heuristic function h(n)**.$$h(n) = \text{estimated cost of the cheapest path from the state at node n to a goal state}$$
##### Greedy best-first search
- **Process**:
	- tries to expand node closest to the goal believing its likely to lead to a solution quickly
$$f(n) = h(n)$$
- **Properties**:
    - **Complete**: incomplete even in a finite state space. If any step in the solution is farther than the current node, it will never make that step and is subject to infinite loops.
    - **Optimal**: nonoptimal because it only tries to get the closest at every step
    - **Time complexity**: $O(b^m)$ where m is the max depth of any node.
    - **Space complexity**: $O(b^m)$ where m is the max depth of the search space
    - NOTE: with a good heuristic function, the complexity can be reduces substantially.
	
##### Greedy best-first search
- **Process**:
	- Most widely known. Combines g(n), the cost to reach the node, and h(n) the cost to get from the node to the goal. which gives the estimated cost of the cheapest solution through n.
$$f(n) = g(n) + h(n)$$
	- same process as UCS
- **Properties**:
	- Depending on h(n) (it satisfies some conditions) A* search is complete and optimal
    - **Time complexity**: $O(b^m)$ where m is the max depth of any node.
    - **Space complexity**: $O(b^m)$ where m is the max depth of the search space
    - NOTE: with a good heuristic function, the complexity can be reduces substantially.
    - **Optimal**: tree search is optimal if h(n) is admissible, graph search is optimal when h(n) is consistent
	
- **Conditions for optimality: Admissibility and consistency**:
	- h(n) is admissible: never overestimates the cost to reach the goal, thus f(n) never overestimates the true cost of the solution along the current path through n
	- consistency (only required for graph search): h(n) is consistent if for every node n and every successor n' of n generated by any action a, the estimated cost of reaching the goal from n is no greater than the step cost of getting n' plus the estimated cost of reaching the goal from n'
	$$h(n) <= c(n, a, c') + h(n') $$

#### 4.1 Local Search algorithms and optimization problems
Local search algorithms operate using a single current node (rather than multiple paths) and generally move only to neighbors of that node.

Two advantages: 
1) very little memory (usually constant)
2) can find reasonable solutions in large or infinite state spaces

used for pure optimization problems where the goal is to find the best state according to an objective function.

This chapter has examined search algorithms for problems beyond the “classical” case of finding the shortest path to a goal in an observable, deterministic, discrete environment.• Local search methods such as hill climbing operate on complete-state formulations, keeping only a small number of nodes in memory. Several stochastic algorithms have been developed, including simulated annealing, which returns optimal solutions when given an appropriate cooling schedule.
- Many local search methods apply also to problems in continuous spaces. Linear pro-gramming and convex optimization problems obey certain restrictions on the shape of the state space and the nature of the objective function, and admit polynomial-time algorithms that are often extremely efficient in practice.
- A genetic algorithm is a stochastic hill-climbing search in which a large population of states is maintained. New states are generated by mutation and by crossover, which combines pairs of states from the population.

Section 5.9. Summary 189
would not want alpha–beta to waste time determining a precise value for the lone good move.
Better to just make the move quickly and save the time for later. This leads to the idea of the
utility of a node expansion. A good search algorithm should select node expansions of high
utility—that is, ones that are likely to lead to the discovery of a significantly better move. If
there are no node expansions whose utility is higher than their cost (in terms of time), then
the algorithm should stop searching and make a move. Notice that this works not only for
clear-favorite situations but also for the case of symmetrical moves, for which no amount of
search will show that one move is better than another.
This kind of reasoning about what computations to do is called metareasoning (rea-METAREASONING
soning about reasoning). It applies not just to game playing but to any kind of reasoning
at all. All computations are done in the service of trying to reach better decisions, all have
costs, and all have some likelihood of resulting in a certain improvement in decision quality.
Alpha–beta incorporates the simplest kind of metareasoning, namely, a theorem to the effect
that certain branches of the tree can be ignored without loss. It is possible to do much better.
In Chapter 16, we see how these ideas can be made precise and implementable.
Finally, let us reexamine the nature of search itself. Algorithms for heuristic search
and for game playing generate sequences of concrete states, starting from the initial state
and then applying an evaluation function. Clearly, this is not how humans play games. In
chess, one often has a particular goal in mind—for example, trapping the opponent’s queen—
and can use this goal to selectively generate plausible plans for achieving it. This kind of
goal-directed reasoning or planning sometimes eliminates combinatorial search altogether.
David Wilkins’ (1980) PARADISE is the only program to have used goal-directed reasoning
successfully in chess: it was capable of solving some chess problems requiring an 18-move
combination. As yet there is no good understanding of how to combine the two kinds of
algorithms into a robust and efficient system, although Bridge Baron might be a step in the
right direction. A fully integrated system would be a significant achievement not just for
game-playing research but also for AI research in general, because it would be a good basis
for a general intelligent agent.

#### 5.9 SUMMARY
We have looked at a variety of games to understand what optimal play means and to understand how to play well in practice. The most important ideas are as follows:
- A game can be defined by the initial state (how the board is set up), the legal actions in each state, the result of each action, a terminal test (which says when the game is over), and a utility function that applies to terminal states.
- In two-player zero-sum games with perfect information, the minimax algorithm can select optimal moves by a depth-first enumeration of the game tree.
- The alpha–beta search algorithm computes the same optimal move as minimax, but achieves much greater efficiency by eliminating subtrees that are provably irrelevant.
- Usually, it is not feasible to consider the whole game tree (even with alpha–beta), so we need to cut the search off at some point and apply a heuristic evaluation function that estimates the utility of a state.
- Many game programs precompute tables of best moves in the opening and endgame so that they can look up a move rather than search.
- Games of chance can be handled by an extension to the minimax algorithm that evaluates a chance node by taking the average utility of all its children, weighted by the probability of each child.
- Optimal play in games of imperfect information, such as Kriegspiel and bridge, requires reasoning about the current and future belief states of each player. A simple approximation can be obtained by averaging the value of an action over each possible configuration of missing information.
- Programs have bested even champion human players at games such as chess, checkers, and Othello. Humans retain the edge in several games of imperfect information, such as poker, bridge, and Kriegspiel, and in games with very large branching factors and little good heuristic knowledge, such as Go.

#### 6 CSP
- Constraint satisfaction problems (CSPs) represent a state with a set of variable/value pairs and represent the conditions for a solution by a set of constraints on the variables. Many important real-world problems can be described as CSPs.
- A number of inference techniques use the constraints to infer which variable/value pairs are consistent and which are not. These include node, arc, path, and k-consistency.
- Backtracking search, a form of depth-first search, is commonly used for solving CSPs. Inference can be interwoven with search.
- The minimum-remaining-values and degree heuristics are domain-independent methods for deciding which variable to choose next in a backtracking search. The least constraining-value heuristic helps in deciding which value to try first for a given variable. Backtracking occurs when no legal assignment can be found for a variable. Conflict-directed backjumping backtracks directly to the source of the problem.
- Local search using the min-conflicts heuristic has also been applied to constraint satis- faction problems with great success.
- The complexity of solving a CSP is strongly related to the structure of its constraint graph. Tree-structured problems can be solved in linear time. Cutset conditioning can reduce a general CSP to a tree-structured one and is quite efficient if a small cutset can be found. Tree decomposition techniques transform the CSP into a tree of subproblems and are efficient if the tree width of the constraint graph is small.

### Chapter 4: Beyond Classical Search

#### 4.1 Local Search Algorithms

**Local search algorithms** operate over the space of states rather than paths and are typically used for optimization problems. They keep only a small number of states in memory, focusing on finding a good enough solution rather than exploring all possibilities.

**Hill Climbing**

- **Process**:
    - Starts from an initial state and iteratively moves to the neighbor with the highest value according to the objective function.
    - Continues until it reaches a peak (local maximum) or a state where no better neighbors exist.
- **Types**:
    - **Simple Hill Climbing**: Examines only one neighbor at a time.
    - **Steepest-Ascent Hill Climbing**: Examines all neighbors and chooses the one with the highest value.
- **Properties**:
    - **Not complete**: Can get stuck in local maxima.
    - **Not optimal**: May not find the global maximum.
    - **Time complexity**: Depends on the number of neighbors and the size of the search space.
    - **Space complexity**: Low, as it only stores the current state and its neighbors.

**Simulated Annealing**

- **Process**:
    - A probabilistic technique that attempts to avoid local maxima by allowing moves to worse states with decreasing probability over time (controlled by a temperature parameter).
    - The temperature decreases according to a cooling schedule, which determines the rate at which the probability of accepting worse states declines.
- **Properties**:
    - **Complete**: Given an appropriate cooling schedule, it will eventually find the global optimum.
    - **Optimal**: Can find the global optimum with a high probability.
    - **Time complexity**: Depends on the cooling schedule and problem size.
    - **Space complexity**: Low, as it stores only the current state and its neighbors.

#### 4.2 Search in Continuous Spaces

**Linear Programming**

- **Process**:
    - Used to optimize a linear objective function subject to linear constraints.
    - Formulated as: Maximize or minimize cTxc^T xcTx subject to Ax≤bAx \leq bAx≤b and x≥0x \geq 0x≥0.
- **Properties**:
    - **Polynomial-time solvable**: Efficient algorithms like the Simplex method or Interior Point methods exist.
    - **Applications**: Resource allocation, scheduling, and network flow problems.

**Convex Optimization**

- **Process**:
    - Focuses on optimizing a convex objective function subject to convex constraints.
    - Formulated as: Minimize f(x)f(x)f(x) subject to gi(x)≤0g_i(x) \leq 0gi​(x)≤0 and hj(x)=0h_j(x) = 0hj​(x)=0, where fff, gig_igi​, and hjh_jhj​ are convex functions.
- **Properties**:
    - **Polynomial-time solvable**: Efficient algorithms like Gradient Descent or Quadratic Programming are used.
    - **Applications**: Machine learning, finance, and engineering problems where the objective function and constraints are convex.

#### 4.3 Genetic Algorithms

**Genetic Algorithm**

- **Process**:
    - Mimics natural evolution by maintaining a population of states (chromosomes).
    - **Operators**:
        - **Mutation**: Randomly alters parts of a chromosome to maintain genetic diversity.
        - **Crossover**: Combines parts of two parent chromosomes to create offspring.
    - **Selection**: Chooses the fittest individuals to create new generations.
- **Properties**:
    - **Not complete**: Does not guarantee finding the global optimum but can find good solutions in practice.
    - **Optimal**: Depends on the problem and the parameters used.
    - **Time complexity**: Depends on population size, number of generations, and problem size.
    - **Space complexity**: Moderate, as it maintains a population of states.

### Chapter 5: Adversarial Search and Games

#### 5.1 Introduction to Adversarial Search

**Adversarial search** deals with environments where agents compete against each other, often with conflicting goals. These problems are common in game theory, where each player tries to maximize their own benefit while minimizing the benefit of their opponent.

- **Two-player games**: Classic examples include chess, checkers, and tic-tac-toe.
- **Game theory**: Analyzes strategies where the outcome depends on the choices of multiple agents.

#### 5.2 Minimax Algorithm

**Minimax Algorithm**

- **Process**:
    - Used in two-player, zero-sum games where one player's gain is another player's loss.
    - **Objective**: Maximize the minimum gain (minimax strategy) to minimize the maximum possible loss.
    - **Steps**:
        1. **Generate the game tree**: Each node represents a state of the game, and each edge represents a possible move.
        2. **Evaluate terminal nodes**: Assign a value to each terminal state based on the outcome (win, lose, draw).
        3. **Propagate values back up the tree**: Use the minimax rule:
            - **Maximizing Player**: Chooses the move with the highest value.
            - **Minimizing Player**: Chooses the move with the lowest value.
- **Properties**:
    - **Complete**: Finds the optimal move assuming both players play optimally.
    - **Optimal**: Guarantees the best outcome for the maximizing player given optimal play by both players.
    - **Time complexity**: O(bd)O(b^d)O(bd), where bbb is the branching factor and ddd is the depth of the game tree.
    - **Space complexity**: O(bd)O(b^d)O(bd), as it stores the entire game tree.

#### 5.3 Alpha-Beta Pruning

**Alpha-Beta Pruning**

- **Process**:
    - Optimizes the minimax algorithm by pruning branches that cannot influence the final decision, reducing the number of nodes evaluated.
    - **Parameters**:
        - **Alpha**: The best value found so far for the maximizing player.
        - **Beta**: The best value found so far for the minimizing player.
    - **Steps**:
        1. **Traversal**: Search the game tree like minimax.
        2. **Prune**: Stop exploring a branch if it cannot affect the final decision.
            - **Alpha Cutoff**: If a node’s value is less than or equal to the alpha value, prune it.
            - **Beta Cutoff**: If a node’s value is greater than or equal to the beta value, prune it.
- **Properties**:
    - **Complete**: Finds the optimal move as long as the entire game tree is traversed.
    - **Optimal**: Guarantees the same result as minimax with fewer nodes evaluated.
    - **Time complexity**: Best case is O(bd/2)O(b^{d/2})O(bd/2), where bbb is the branching factor and ddd is the depth of the game tree.
    - **Space complexity**: O(bd)O(b^d)O(bd) due to storing the game tree.

#### 5.4 Games with Imperfect Information

**Imperfect Information Games**

- **Process**:
    - Games where players do not have complete knowledge of the state of the game.
    - Examples include poker and bridge.
- **Strategies**:
    - **Probabilistic Models**: Use probability distributions to represent uncertain information.
    - **Belief States**: Maintain a probability distribution over possible states rather than a single state.
- **Algorithms**:
    - **Monte Carlo Tree Search (MCTS)**: Uses random sampling to estimate the value of moves and prune the search space.
        - **Steps**:
            1. **Selection**: Traverse the tree to select a node.
            2. **Expansion**: Add new nodes.
            3. **Simulation**: Play out a random game from the new node.
            4. **Backpropagation**: Update the values of nodes based on the result of the simulation.
        - **Properties**:
            - **Not complete**: Depends on the number of simulations.
            - **Optimal**: Can find good solutions with enough simulations.
            - **Time complexity**: Depends on the number of simulations.
            - **Space complexity**: Moderate, based on the size of the search tree.

#### 5.5 Example: Chess

**Chess**

- **Search Complexity**:
    - **Game Tree Size**: Extremely large, making exhaustive search impractical.
    - **Heuristic Evaluation**: Instead of evaluating all possible moves, use heuristics to evaluate board positions.
- **Algorithms**:
    - **Minimax with Alpha-Beta Pruning**: Commonly used in chess engines to make optimal moves efficiently.
    - **Heuristic Functions**: Evaluate board positions based on material count, piece positions, and control of the board.


### Chapter 6: Constraint Satisfaction Problems

#### 6.1 Introduction to Constraint Satisfaction Problems (CSPs)

**Constraint Satisfaction Problems (CSPs)** involve finding a solution to a set of variables subject to constraints. Each variable must be assigned a value from a domain such that all constraints are satisfied.

- **Components**:
    - **Variables**: The elements that need to be assigned values.
    - **Domains**: The possible values that each variable can take.
    - **Constraints**: Restrictions on the values that variables can simultaneously take.

**Example Problems**:

- **N-Queens Problem**: Place N queens on an N×N chessboard so that no two queens threaten each other.
- **Sudoku**: Fill a 9×9 grid so that each column, row, and 3×3 subgrid contains all digits from 1 to 9.

#### 6.2 Constraint Propagation

**Constraint Propagation**

- **Process**:
    - Reduces the domain of variables by applying constraints and removing inconsistent values.
    - **Arc Consistency**: A constraint is arc-consistent if for every value of a variable, there is a compatible value for each related variable.
        - **AC-3 Algorithm**: A common algorithm used to enforce arc consistency.
            - **Steps**:
                1. **Initialize**: Add all arcs to a queue.
                2. **Process**: Remove arcs from the queue and revise them.
                3. **Revise**: Update the domain of a variable if necessary and add related arcs to the queue.
            - **Properties**:
                - **Time complexity**: O(ed3)O(ed^3)O(ed3), where eee is the number of constraints, and ddd is the maximum domain size.
                - **Space complexity**: O(ed)O(ed)O(ed).

#### 6.3 Backtracking Search

**Backtracking Search**

- **Process**:
    - A depth-first search approach where variables are assigned values one by one.
    - If a variable assignment leads to a conflict, the algorithm backtracks and tries a different value.
- **Properties**:
    - **Complete**: Will find a solution if one exists, given enough time.
    - **Optimal**: Not guaranteed to find the best solution.
    - **Time complexity**: Exponential in the number of variables.
    - **Space complexity**: Linear in the depth of the search tree.

**Improvements**:

- **Forward Checking**: Reduces the domain of unassigned variables when a variable is assigned a value.
    - **Process**: Checks future variable assignments to prune values that are inconsistent with the current assignment.
- **Constraint Learning**: Memorizes constraints violated during the search to avoid repeating the same mistakes.
- **Dynamic Variable Ordering**: Selects the variable with the fewest remaining values to assign next, improving efficiency.

#### 6.4 Heuristic Approaches

**Heuristic Approaches**

- **Process**:
    - **Variable Ordering**: Choose variables with the smallest domain or highest constraint tightness to assign first.
        - **Degree Heuristic**: Chooses the variable involved in the largest number of constraints.
        - **Least Constraining Value**: Chooses values that leave the most options open for other variables.
- **Properties**:
    - **Time complexity**: Depends on the heuristics used and the problem size.
    - **Space complexity**: Depends on the search strategy and heuristics.

#### 6.5 Constraint Satisfaction in Practice

**Applications**:

- **Scheduling**: Assign times to events or resources without conflicts.
- **Timetabling**: Allocate resources to tasks (e.g., exam schedules) while respecting constraints.
- **Resource Allocation**: Distribute resources (e.g., tasks to workers) to meet constraints and optimize objectives.

**Techniques**:

- **Constraint Programming**: A paradigm where problems are expressed in terms of constraints and solved using specialized solvers.
- **SAT Solvers**: Convert CSPs into Boolean satisfiability problems and solve them using SAT solving techniques.

**Example**: Sudoku Solver

- **CSP Formulation**:
    - **Variables**: Cells in the grid.
    - **Domains**: Digits 1 to 9.
    - **Constraints**: Each row, column, and 3×3 subgrid must contain all digits exactly once.
- **Solution**:
    - **Backtracking**: Recursively assign digits to cells, backtracking when constraints are violated.
    - **Constraint Propagation**: Use techniques like AC-3 to reduce the domain of variables.
