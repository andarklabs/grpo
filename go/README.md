goal: get this model file to just use GRPO.

1. make policy head work with grpo
2. replace file `model_pytorch` in the Katago synchronous_loop.sh reference
    - maybe we need a whole new model file ourself 
3. train it with selfplay
4. evaluate against reference
5. win life