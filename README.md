
The graph structure of YAGO3 looks closer to a scale-free network than to a uniformly random graph.
![in degree](img/degree_distribution_in.png)
![out degree](img/degree_distribution_out.png)

But: YAGO3 is as easy to memorize as a uniformly random graph:

![yago3 p@10](img/memorize_triples_05.yago3_10.rnnpathmemory.p@10.n_layers_1.lines.png)

One problem: low accuracy due to ambiguous paths (different objects possible for same subject and predicate)

![yago3 acc](img/memorize_triples_05.yago3_10.rnnpathmemory.acc.n_layers_1.lines.png)
