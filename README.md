<p align="center">
    <img width="200" src="https://i.imgur.com/6Rf2GcE.png">
</p>
<!--- https://i.imgur.com/6Rf2GcE.png --->

# LCRL
Logically-Constrained Reinforcement Learning (LCRL) is a model-free reinforcement learning framework to synthesise
policies for unknown, continuous-state-action Markov Decision Processes (MDPs) under a given Linear Temporal Logic
(LTL) property. LCRL automatically shapes a synchronous reward function on-the-fly. This enables any
off-the-shelf RL algorithm to synthesise policies that yield traces which probabilistically satisfy the LTL property. LCRL produces policies that are certified to satisfy the given LTL property with maximum probability.

## Publications
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
To get the latest version of LCRL, you can clone this repository and install the dependencies:
```
git clone https://github.com/grockious/lcrl.git
cd lcrl
pip3 install .
```

Alternatively, you can directly use
```
pip3 install lcrl
```
or
```
pip3 install git+https://github.com/grockious/lcrl.git
```

## Usage
#### Training an RL agent under an LTL property
A sample training command is:
```
python3 train.py --env 'SlipperyGrid' --layout 'layout_1' --property 'g1-then-g2'
```
where option `--env` specifies an environment object from the subdirectory
```
./environments
```
The option `--layout` determines atomic proposition mapping within the environment
(the environments in `./environments` provide one or more layouts),
and the option `--property` specifies the LTL property.

Use the `-h` option for help and to get a list of the available parameters.
#### Applying LCRL to a black-box MDP and custom LTL property
#### - MDP:
LCRL can be connected to a black-box MDP object that is fully unknown to
the tool. This includes the size of the state space as LCRL automatically keeps track of visited states. For examples of MDP classes
please refer to `./environments`. The MDP object, call it `MDP`, should at
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
The synthesised LDBA can be used as an object of the class `./automata/ldba.py`.  

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
to update the accepting frontier set.

## Reference
Please use this bibtex entry if you want to cite this repository in your publication:

```
@misc{lcrl_repo,
  author = {Mohammadhosein Hasanbeig, Alessandro Abate, and Daniel Kroening},
  title = {Logically-Constrained Reinforcement Learning Code Repository},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/grockious/lcrl}},
}
```

## License
This project is licensed under the terms of the [MIT License](/LICENSE)
