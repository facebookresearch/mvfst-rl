# torchbeast
A PyTorch implementation of [IMPALA: Scalable Distributed
Deep-RL with Importance Weighted Actor-Learner Architectures
by Espeholt, Soyer, Munos et al](https://arxiv.org/abs/1802.01561). More details in [TorchBeast paper](https://arxiv.org/abs/1910.03552).

Stripped down and modified from https://github.com/facebookresearch/torchbeast.
The primary differences from torchbeast master are:
- Only polybeast with pure C++ rpcenv and actorpool implementation, and
  environment <-> RL agent interaction via RPC. No monobeast, gym, atari wrappers, etc.
- Inverted gRPC client-server setup with ActorPool as the server and environment
  as the client. This simplifies server threading logic (abstracted by gRPC), allows
  easy scaling to arbitrary number of parallel actors, and makes it possible for clients
  behind a NAT (such as in [Pantheon](https://pantheon.stanford.edu)) to initiate
  connections to RL agent server.
- Ability to handle partial rollouts (less than the specified unroll length)
  as a way to handle episodic training with `env.reset()` implemented as
  network tear-down and reinitialization.

## (Very rough) overview

```
|-----------------|     |-----------------|                  |-----------------|
|     ACTOR 1     |     |     ACTOR 2     |                  |     ACTOR n     |
|-------|         |     |-------|         |                  |-------|         |
|       |  .......|     |       |  .......|     .   .   .    |       |  .......|
|  Env  |<-.Model.|     |  Env  |<-.Model.|                  |  Env  |<-.Model.|
|       |->.......|     |       |->.......|                  |       |->.......|
|-----------------|     |-----------------|                  |-----------------|
   ^     I                 ^     I                              ^     I
   |     I                 |     I                              |     I Actors
   |     I rollout         |     I rollout               weights|     I send
   |     I                 |     I                     /--------/     I rollouts
   |     I          weights|     I                     |              I (frames,
   |     I                 |     I                     |              I  actions
   |     I                 |     v                     |              I  etc)
   |     L=======>|--------------------------------------|<===========J
   |              |.........      LEARNER                |
   \--------------|..Model.. Consumes rollouts, updates  |
     Learner      |.........       model weights         |
      sends       |--------------------------------------|
     weights
```

Actors generate rollouts (`T` steps of data from environment-agent interactions,
includes environment frames, agent actions and policy logits, and other data).

The learner consumes that experience, computes a loss and updates the weights.
