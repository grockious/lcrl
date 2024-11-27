import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='lcrl',
    version='0.0.9.2',
    author="Hosein Hasanbeig",
    author_email="hosein.hasanbeig@icloud.com",
    keywords='rl, logic, environment, agent',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/grockious/lcrl',
    description='Logically-Constrained Reinforcement Learning',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        'tensorflow==2.11.1',
        'numpy',
        'matplotlib',
        'dill>=0.3.2',
        'imageio',
        'tqdm'
    ]
)
