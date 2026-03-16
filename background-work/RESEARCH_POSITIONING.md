# Research Positioning

This experiment is not a direct trained-AttnRes replication because we do not yet have a practical local trained-routing comparison at the scale we can run here. Its position is narrower and cleaner:

- infer what latent depth-routing structure or effective depth mixture can be recovered from standard transformer representations
- test whether those recovered patterns match the published Figure 8 pattern surface reported for trained AttnRes models
- show where existing interpretability tools do not directly recover input-dependent routing weights and would need adaptation under dynamic routing
- connect the recovered routing structure to safety-relevant features such as refusal or honesty

The strongest publishable framing is:

`We show that standard transformers contain a structured effective depth mixture or latent routing signal that existing interpretability tools do not directly recover as input-dependent routing weights.`

That is different from claiming:

- that trained AttnRes behavior has been directly reproduced
- that co-adapted routing and computation have been measured
- that AttnRes is empirically proven more interpretable in deployed models
