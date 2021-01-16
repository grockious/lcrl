from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='lcrl',
    version='0.0.2',
    author="Hosein Hasanbeig",
    author_email="hosein.hasanbeig@cs.ox.ac.uk",
    keywords='rl, logic, environment, agent',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/grockious/lcrl',
    description='Logically-Constrained Reinforcement Learning',
    packages=['automata', 'environments', 'scripts'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
        'numpy',
        'matplotlib',
        'dill>=0.3.2',
        'imageio',
        'tqdm'
    ]
)


