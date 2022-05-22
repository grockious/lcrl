import math
import numpy as np


class CartPole:
    """
    CartPole environment modelled as an MDP

    ...

    Attributes
    ----------
    gravity: float
        gravitational acceleration
    masscart: float
        mass of the cart
    masspole: float
        mass of the pole
    length: float
        length of the pole
    force_mag: float
        magnitude of the applied force

    Methods
    -------
    reset()
        resets the MDP state
    step(action)
        changes the state of the MDP upon executing an action, where the action set is {right,up,left,down,stay}
    state_label(state)
        outputs the label of input state
    """

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02
        self.kinematics_integrator = 'euler'
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.current_state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.action_space = [-1, 1]

    def step(self, action):
        x, x_dot, theta, theta_dot = self.current_state
        force = self.force_mag if action[0] > 0 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.current_state = np.array((x, x_dot, theta, theta_dot))

        return self.current_state

    def reset(self):
        self.current_state = np.random.uniform(low=-0.05, high=0.05, size=(4,))

    def state_label(self, state):
        x, x_dot, theta, theta_dot = state
        if x < -self.x_threshold or \
                x > self.x_threshold or \
                theta < -self.theta_threshold_radians or \
                theta > self.theta_threshold_radians:
            return 'd'
        else:
            return 'u'
