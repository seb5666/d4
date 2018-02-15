# Differentiable symbolic policy for black

I have implemented the REINFORCE algorithm [link](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume4/kaelbling96a-html/node37.html)
to train the policy (see sketches/simple_policy.py).

I haven't optimised the hyper-parameters much, but the current ones seem to work, making the policy converge on choosing 14 as a threshold. 
The policy has the form
```bash
if sum < threshold:
    hit
 else:
    stick
```
where the threshold is learned by the very simple neural network. In this case it is really just learning weights that are then 
softmaxed to produce probabilities for each threshold. The comparison is then done in a soft way, using the differentiable stack machine
which gives a probability distribution over the two actions: hit and stick.

## Installation
I have been running this with Python 3.5.
In addition, you will need the following libraries:
- Open AI Gym
    ```bash
    pip install gym
    ```
- Tensorflow (version 1.4.1)
    ```bash
    pip install tensorflow==1.4.1
    ```
- matplotlib
    ```bash
    pip install matplotlib
    ```i
- Funcparserlib
    ```bash
    pip install funcparserlib
    ```
- Scipy
    ```bash
    pip install scipy
    ```
I highly recommend using a virtual environment for this.

## Running
There is no need to run this on a GPU as there are very few weights 
(only as many as the number of choices for the threshold).

To run the training script, run from the main directory:
```bash
python blackjack/train.py
```

There might be some issues with the module system in Python not finding the imported files. In this case you need to add
the blackjack directory to the PYTHONPATH directly:
```bash
cd blackjack/
export PYTHONPATH=.case
cd ..
```

After training is completed, there should be 4 graphs appearing:
1. The policy before training started. This is randomly intitialised and will have some threshold somewhere.
The usable ace and no usable ace cases are identical as the policy doesn't consider this feature. It only considers the 
player's sum.
2. The training weights: How the weights of the neural network change during training. At the end the weight corresponding
to the choice of 14 should be the highest
3. The average reward received over time. These rewards are computed using the argmaxed i.e. fully deterministic policy.
4. The policy after training is completed


## Next steps
- It is very easy for this case to obtain a fully symbolic policy as the policy is only dependent on the player's sum. The
neural network is not dependent on anything (static encoder in policy). When the policy will depend on the last few elements
on the data stack it will be harder to retrieve it. 
- As Jason mentioned, during test time we can retrieve the actual forth words selected by the neural network but I'm not sure
this will allow us to generate a fully symbolic policy as the selected forth word will depend on the input data, hence might be
different for every input and therefore be no more interpretable than just querying a standard neural net with all input 
combinations. Finding out how to do this (if possible) should be the next priority.
- I am only really using the data stack of the differentiable forth machine. The heap isn't utilized at all and might be 
an overhead that is not needed for the kind of policies we want to train.