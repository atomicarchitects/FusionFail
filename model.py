
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import e3nn_jax as e3nn

class TensorProduct(nn.Module):

    @nn.compact
    def __call__(self, x: e3nn.IrrepsArray, y: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        return e3nn.tensor_product(x, y)


class Layer(nn.Module):
    target_irreps: e3nn.Irreps
    denominator: float
    sh_lmax: int

    @nn.compact
    def __call__(self, graphs, positions):
        target_irreps = e3nn.Irreps(self.target_irreps)

        def update_edge_fn(edge_features, sender_features, receiver_features, globals):
            sh = e3nn.spherical_harmonics(
                list(range(1, self.sh_lmax + 1)),
                positions[graphs.receivers] - positions[graphs.senders],
                True,
            )
            sender_features = e3nn.as_irreps_array(sender_features)

            tp = TensorProduct()(sender_features, sh)

            return e3nn.concatenate(
                [sender_features, tp]
            ).regroup()

        def update_node_fn(node_features, sender_features, receiver_features, globals):
            node_feats = receiver_features / self.denominator
            node_feats = e3nn.flax.Linear(target_irreps, name="linear_pre")(node_feats)
            node_feats = e3nn.scalar_activation(node_feats)
            node_feats = e3nn.flax.Linear(target_irreps, name="linear_post", force_irreps_out=True)(node_feats)
            shortcut = e3nn.flax.Linear(
                node_feats.irreps, name="shortcut", force_irreps_out=True
            )(node_features)
            return shortcut + node_feats

        return jraph.GraphNetwork(update_edge_fn, update_node_fn)(graphs)


class NequIP(nn.Module):
    @nn.compact
    def __call__(self, graphs):
        positions = e3nn.IrrepsArray("1o", graphs.nodes)
        graphs = graphs._replace(nodes=jnp.ones((len(positions), 1)))

        layers = 2 * ["32x0e + 32x0o + 8x1e + 8x1o + 8x2e + 8x2o"] + ["0o + 7x0e"]
        for irreps in layers:
            graphs = Layer(irreps, denominator=1.5, sh_lmax=3)(graphs, positions)

        # Readout logits
        pred = e3nn.scatter_sum(
            graphs.nodes.array, nel=graphs.n_node
        )  # [num_graphs, 1 + 7]
        odd, even1, even2 = pred[:, :1], pred[:, 1:2], pred[:, 2:]
        logits = jnp.concatenate([odd * even1, -odd * even1, even2], axis=1)
        assert logits.shape == (len(graphs.n_node), 8), logits.shape  # [num_graphs, num_classes]

        return logits