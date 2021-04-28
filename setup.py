import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lcrl',
    version='0.0.5',
    author="Hosein Hasanbeig",
    author_email="hosein.hasanbeig@cs.ox.ac.uk",
    keywords='rl, logic, environment, agent',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/grockious/lcrl',
    description='Logically-Constrained Reinforcement Learning',
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'matplotlib',
        'dill>=0.3.2',
        'imageio',
        'tqdm',
        'tensorflow>=2'
    ]
)
