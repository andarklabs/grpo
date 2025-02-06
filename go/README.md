goal: get this model file to just use GRPO.

1. make policy head work with grpo
2. replace file `model_pytorch` in the Katago synchronous_loop.sh reference
    - maybe we need a whole new model file ourself. Yes, we may need to do this. For now I'm hoping that the surgery works (their file is actually not as chaotic as I first thought. As long as we understand the model class we should be fine)
3. train it with selfplay
    - We also need to change the way that it trains in order for it to be used with grpo
4. evaluate against reference
5. win life

use existing model checkpoint with synchronous loop: train from a previous checkpoint

grpo surgery 
    - determine what the policy head gets as input (shape and information)
    - determine the same for value head
    - look into weights file to see what goes to what part 
        - Parts:
            - 2 conv layers + 1 dense layer
            - Res trunk (made of 18 blocks - conv, resbottleneck, some global pooling)
            - bias/norm
            - policy
            - value
            --- intermediate heads ---
            - norm 
            - policy
            - value

grpo paper: https://arxiv.org/abs/2402.03300