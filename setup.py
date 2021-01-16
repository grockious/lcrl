from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='lcrl',
    version='0.0.2',
    keywords='rl, logic, environment, agent',
    url='https://github.com/grockious/lcrl',
    description='Logically-Constrained Reinforcement Learning',
    packages=['automata', 'environments', 'scripts'],
    python_requires='>=3.5',
    install_requires=[
        'numpy',
        'matplotlib',
        'dill>=0.3.2',
        'imageio',
        'tqdm'
    ]
)


