from setuptools import setup

setup(
    name='lcrl',
    version='0.0.1',
    keywords='rl, logic, environment, agent',
    url='https://github.com/grockious/lcrl',
    description='Logically-Constrained Reinforcement Learning',
    packages=['automata', 'environments', 'scripts'],
    python_requires='>=3.5',
    install_requires=[
        'numpy',
        'matplotlib',
        'dill>=0.3.2'
    ]
)
