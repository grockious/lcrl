import os
import random
from operator import add
import numpy as np


class PacMan:
    """
    Arcade Pacman modelled as an MDP

    ...

    Attributes
    ----------
    layout: string
        path to a layout file

    Methods
    -------
    process_layout()
        creates a labelling function and its associated graph from a '.py' file
        each character in a '.py' file represents a different type of object:
         % - Wall
         . - Token
         o - Food
         G - Ghost
         P - Pacman
        Other characters are ignored
        For built-in examples see `./environments/layouts/`
    reset()
        resets the MDP state, including ghosts and agent
    step(action)
        changes the state of the MDP upon executing an action, where the action set is {right,up,left,down,stay}
    state_label(state)
        outputs the label of input state
    state2cell(state)
        converts the agent / ghost state to its cell number in the grid
    cell2state(int)
        converts a cell number to coordinates of that cell
    """

    def __init__(
            self,
            layout='small'
    ):
        file = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'layouts', layout) + '.py')
        self.layoutText = [line.strip() for line in file]
        file.close()
        self.width = len(self.layoutText[0])
        self.height = len(self.layoutText)
        self.labels = np.empty([self.width, self.height], dtype=object)
        self.agent_state = []
        self.ghosts_state = []
        self.mdp_graph = {}
        self.agent_initial_state = []
        self.ghosts_initial_state = []
        self.current_state = []
        # directional actions
        self.action_space = [
            "right",
            "up",
            "left",
            "down",
            "stay"
        ]
        self.process_layout()

    def process_layout(self):
        food_counter = 0
        max_y = self.height - 1
        for y in range(self.height):
            for x in range(self.width):
                layout_char = self.layoutText[max_y - y][x]
                if layout_char == '%':
                    self.labels[x][y] = 'obstacle'
                elif layout_char == '.':
                    self.labels[x][y] = 'token'
                elif layout_char == 'o':
                    food_counter += 1
                    self.labels[x][y] = 'food' + str(food_counter)
                elif layout_char == 'P':
                    self.agent_state = self.location2cell([x, y])
                    self.agent_initial_state = self.location2cell([x, y])
                elif layout_char in ['G']:
                    self.ghosts_state.append(self.location2cell([x, y]))
                    self.ghosts_initial_state.append(self.location2cell([x, y]))
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                graph_vertex = self.location2cell([x, y])
                self.mdp_graph[graph_vertex] = []
                for a in self.action_space:
                    self.agent_state = self.location2cell([x, y])
                    neightbour_cell = self.step(a, False)
                    if neightbour_cell[0] not in self.mdp_graph[graph_vertex]:
                        self.mdp_graph[graph_vertex].append(neightbour_cell[0])
        self.reset()

    def reset(self):
        self.agent_state = self.agent_initial_state
        self.ghosts_state = self.ghosts_initial_state.copy()
        self.current_state = [self.agent_state] + self.ghosts_state

    def step(self, action, ghosts_moving=1):
        # agent movement dynamics:
        if action == 'right':
            agent_next_location = list(map(add, self.cell2location(self.agent_state), [0, 1]))
        elif action == 'up':
            agent_next_location = list(map(add, self.cell2location(self.agent_state), [-1, 0]))
        elif action == 'left':
            agent_next_location = list(map(add, self.cell2location(self.agent_state), [0, -1]))
        elif action == 'down':
            agent_next_location = list(map(add, self.cell2location(self.agent_state), [1, 0]))
        elif action == 'stay':
            agent_next_location = self.cell2location(self.agent_state)

        # check for obstacles
        if self.labels[agent_next_location[0]][agent_next_location[1]] is not None \
                and 'obstacle' in self.labels[agent_next_location[0]][agent_next_location[1]]:
            agent_next_location = self.cell2location(self.agent_state)

        # update agent state
        self.agent_state = self.location2cell(agent_next_location)

        # how ghosts chase the agent
        if ghosts_moving:
            for i in range(len(self.ghosts_state)):
                # with probability of 40% the ghosts chase the pacman
                if random.random() < 0.4:
                    try:
                        self.ghosts_state[i] = \
                            self.find_shortest_path(self.mdp_graph,
                                                    self.ghosts_state[i],
                                                    self.agent_state)[1]
                    except IndexError:
                        self.ghosts_state[i] = \
                            self.find_shortest_path(self.mdp_graph,
                                                    self.ghosts_state[i],
                                                    self.agent_state)[0]
                # with probability of 60% the ghosts take a random action
                else:
                    ghost_i_next_location = list(map(add, self.cell2location(self.ghosts_state[i]),
                                                     random.sample([[1, 0],
                                                                    [-1, 0],
                                                                    [0, -1],
                                                                    [1, 0]], 1)[0]))
                    # check for obstacles
                    if self.labels[ghost_i_next_location[0]][ghost_i_next_location[1]] is not None \
                            and 'obstacle' in self.labels[ghost_i_next_location[0]][ghost_i_next_location[1]]:
                        ghost_i_next_location = self.cell2location(self.ghosts_state[i])

                    # update ghost state
                    self.ghosts_state[i] = self.location2cell(ghost_i_next_location)
        # return the MDP state
        mdp_state = [self.agent_state] + self.ghosts_state
        self.current_state = mdp_state
        return mdp_state

    def state_label(self, state):
        location = self.cell2location(state[0])
        if self.agent_state in self.ghosts_state:
            return 'ghost'
        else:
            return self.labels[location[0]][location[1]]

    def location2cell(self, state):
        return state[1] + state[0] * self.height

    def cell2location(self, cell_num):
        return [int(cell_num / self.height), cell_num % self.height]

    def find_shortest_path(self, graph, start, end, path=None):
        if path is None:
            path = []
        path = path + [start]
        if start == end:
            return path
        if start not in graph.keys():
            return None
        shortest = None
        for node in graph[start]:
            if node not in path:
                newpath = self.find_shortest_path(graph, node, end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest
