<p align="center">
    <img width="250" src="https://raw.githubusercontent.com/grockious/lcrl/master/assets/lcrl.png">
</p>
<!--- https://i.imgur.com/6Rf2GcE.png --->

# LCRL
Logically-Constrained Reinforcement Learning (LCRL) is a model-free reinforcement learning framework to synthesise
policies for unknown, continuous-state-action Markov Decision Processes (MDPs) under a given Linear Temporal Logic
(LTL) property. LCRL automatically shapes a synchronous reward function on-the-fly. This enables any
off-the-shelf RL algorithm to synthesise policies that yield traces which probabilistically satisfy the LTL property. LCRL produces policies that are certified to satisfy the given LTL property with maximum probability.

## Publications
LCRL Tool Paper:
* Hasanbeig, M., Kroening, D., Abate, A., "LCRL: Certified Policy Synthesis via Logically-Constrained Reinforcement Learning", QEST, 2022. [[PDF (soon)]]()

LCRL Foundations:
* Hasanbeig, M. , Jeppu, N. Y., Abate, A., Melham, T., Kroening, D., "DeepSynth: Automata Synthesis for Automatic Task Segmentation in Deep Reinforcement Learning", AAAI Conference on Artificial Intelligence, 2021. [[PDF]](https://arxiv.org/pdf/1911.10244.pdf)
* Hasanbeig, M. , Abate, A. and Kroening, D., "Cautious Reinforcement Learning with Logical Constraints", International Conference on Autonomous Agents and Multi-agent Systems, 2020. [[PDF]](http://ifaamas.org/Proceedings/aamas2020/pdfs/p483.pdf)
* Hasanbeig, M. , Kroening, D. and Abate, A., "Deep Reinforcement Learning with Temporal Logics", International Conference on Formal Modeling and Analysis of Timed Systems, 2020. [[PDF]](https://link.springer.com/content/pdf/10.1007%2F978-3-030-57628-8_1.pdf)
* Hasanbeig, M. , Kroening, D. and Abate, A., "Towards Verifiable and Safe Model-Free Reinforcement Learning", Workshop on Artificial Intelligence and Formal Verification, Logics, Automata and Synthesis (OVERLAY), 2020. [[PDF]](http://ceur-ws.org/Vol-2509/invited.pdf)
* Hasanbeig, M. , Kantaros, Y., Abate, A., Kroening, D., Pappas, G. J., and Lee, I., "Reinforcement Learning for Temporal Logic Control Synthesis with Probabilistic Satisfaction Guarantees", IEEE Conference on Decision and Control, 2019. [[PDF]](https://arxiv.org/pdf/1909.05304.pdf)
* Hasanbeig, M. , Abate, A. and Kroening, D., "Logically-Constrained Neural Fitted Q-Iteration", International Conference on Autonomous Agents and Multi-agent Systems, 2019. [[PDF]](https://arxiv.org/pdf/1809.07823.pdf)
* Lim Zun Yuan, Hasanbeig, M. , Abate, A. and Kroening, D., "Modular Deep Reinforcement Learning with Temporal Logic Specifications", CoRR abs/1909.11591, 2019. [[PDF]](https://arxiv.org/pdf/1909.11591.pdf)
* Hasanbeig, M. , Abate, A. and Kroening, D., "Certified Reinforcement Learning with Logic Guidance", CoRR abs/1902.00778, 2019. [[PDF]](https://arxiv.org/pdf/1902.00778.pdf)
* Hasanbeig, M. , Abate, A. and Kroening, D., "Logically-Constrained Reinforcement Learning", CoRR abs/1801.08099, 2018. [[PDF]](https://arxiv.org/pdf/1801.08099.pdf)

## Installation
You can install LCRL using 
```
pip3 install lcrl
```

Alternatively, you can clone this repository and install the dependencies:
```
git clone https://github.com/grockious/lcrl.git
cd lcrl
pip3 install .
```
or
```
pip3 install git+https://github.com/grockious/lcrl.git
```

## Usage
#### Training an RL agent under an LTL property

Sample training commands can be found under the `./scripts` directory. LCRL consists of three main classes `MDP`, `LDBA` and the `LCRL` core trainer. Inside LCRL the `MDP` state and the `LDBA` state are synchronised, resulting in an on-the-fly product MDP structure.

&nbsp;
<p align="center">
    <img width="650" src="https://raw.githubusercontent.com/grockious/lcrl/master/assets/lcrl_overview.png">
</p>
<!--- https://i.imgur.com/uH481P0.png --->
&nbsp;

On the product MDP, LCRL automatically shapes a reward function based on the `LDBA` object. An optimal stationary Markov policy synthesised by LCRL on the product
MDP is guaranteed to induce a finite-memory policy on the original MDP that maximises the probability of satisfying the given LTL property. 

The package includes a number of pre-built `MDP` and `LDBA` class objects. For examples of `MDP` and `LDBA` classes
please refer to `./src/environments` and `./src/automata` respectively. For instance, to train an agent for `minecraft-t1` run:

```
python3
```
```python
>>> # import LCRL code trainer module
>>> from src.train import train
>>> # import the pre-built LDBA for minecraft-t1
>>> from src.automata.minecraft_1 import minecraft_1
>>> # import the pre-built MDP for minecraft-t1
>>> from src.environments.minecraft import minecraft
>>> 
>>> LDBA = minecraft_1
>>> MDP = minecraft
>>> 
>>> # train the agent
>>> task = train(MDP, LDBA,
                     algorithm='ql',
                     episode_num=500,
                     iteration_num_max=4000,
                     discount_factor=0.95,
                     learning_rate=0.9
                     )
```

## Applying LCRL to a black-box MDP and custom LTL property
#### - MDP:
LCRL can be connected to a black-box MDP object that is fully unknown to
the tool. This includes the size of the state space as LCRL automatically keeps track of visited states. The MDP object, call it `MDP`, should at
least have the following methods:
```
MDP.reset()
```
to reset the MDP state,
```
MDP.step(action)
```
to change the state of the MDP upon executing `action`,
```
MDP.state_label(state)
```
to output the label of `state`.

#### - LTL:
The LTL property has to be converted to an LDBA, which is a finite-state machine.
An excellent tool for this is OWL, which you can [try online](https://owl.model.in.tum.de/try/).
The synthesised LDBA can be used as an object of the class `./src/automata/ldba.py`.  

The constructed LDBA, call it `LDBA`, is expected to offer the following methods:
```
LDBA.reset()
```
to reset the automaton state and its accepting frontier function,
```
LDBA.step(label)
```
to change the state of the automaton upon reading `label`,
```
LDBA.accepting_frontier_function(state)
```
to update the accepting frontier set. This method is already included in the class `./src/automata/ldba.py`, thus for a custom `LDBA` object you need to only instance this class and specify the `reset()` and `step(label)` methods.  

## Reference
Please cite our tool paper if you use LCRL in your publications:

```
@inproceedings{lcrl_tool,
title={{LCRL}: Certified Policy Synthesis via Logically-Constrained Reinforcement Learning},
author={Hasanbeig, Mohammadhosein and Kroening, Daniel and Abate, Alessandro},
booktitle={International Conference on Quantitative Evaluation of SysTems},
year={2022},
organization={Springer}
}
```

## License
This project is licensed under the terms of the [MIT License](/LICENSE)
