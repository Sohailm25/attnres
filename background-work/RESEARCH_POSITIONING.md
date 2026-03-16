# Research Positioning

This experiment is not a direct trained-AttnRes replication because no AttnRes weights are available. Its position is narrower and cleaner:

- infer what latent depth-routing structure can be recovered from standard transformer representations
- test whether those recovered patterns match the Figure 8 patterns reported for trained AttnRes models
- show where existing interpretability tools would fail or need adaptation under dynamic routing
- connect the recovered routing structure to safety-relevant features such as refusal or honesty

The strongest publishable framing is:

`We show that standard transformers already contain latent depth-routing structure that existing interpretability tools cannot see.`

That is different from claiming:

- that trained AttnRes behavior has been directly reproduced
- that co-adapted routing and computation have been measured
- that AttnRes is empirically proven more interpretable in deployed models
