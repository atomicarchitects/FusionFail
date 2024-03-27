import time
import jraph
import functools

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import e3nn_jax as e3nn
import optax
import nvtx
from tqdm.auto import tqdm
from model import NequIP

def get_tetris_dataset() -> jraph.GraphsTuple:
    pos = [
        [[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 0]],  # chiral_shape_1
        [[1, 1, 1], [1, 1, 2], [2, 1, 1], [2, 0, 1]],  # chiral_shape_2
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],  # square
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],  # line
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],  # corner
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0]],  # L
        [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 1]],  # T
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]],  # zigzag
    ]
    pos = jnp.array(pos, dtype=jnp.float32)
    labels = jnp.arange(8)

    graphs = []

    for p, l in zip(pos, labels):
        senders, receivers = e3nn.radius_graph(p, 1.1)

        graphs += [
            jraph.GraphsTuple(
                nodes=p.reshape((4, 3)),  # [num_nodes, 3]
                edges=None,
                globals=l[None],  # [num_graphs]
                senders=senders,  # [num_edges]
                receivers=receivers,  # [num_edges]
                n_node=jnp.array([len(p)]),  # [num_graphs]
                n_edge=jnp.array([len(senders)]),  # [num_graphs]
            )
        ]

    return jraph.batch(graphs)




def train(steps=200):
    model = NequIP()

    # Optimizer
    opt = optax.adam(learning_rate=0.01)

    @functools.partial(jax.profiler.annotate_function, name="loss_fn")
    def loss_fn(params, graphs):
        logits = model.apply(params, graphs)
        labels = graphs.globals  # [num_graphs]

        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        loss = jnp.mean(loss)
        return loss, logits

    @functools.partial(jax.profiler.annotate_function, name="train_step")
    @jax.jit
    def train_step(params, opt_state, graphs):
        grad_fn = jax.grad(loss_fn, has_aux=True)
        grads, logits = grad_fn(params, graphs)
        labels = graphs.globals
        preds = jnp.argmax(logits, axis=1)
        accuracy = jnp.mean(preds == labels)

        updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, accuracy, preds

    # Dataset
    graphs = get_tetris_dataset()

    # Init
    init = jax.jit(model.init)
    params = init(jax.random.PRNGKey(3), graphs)
    opt_state = opt.init(params)

    # compile jit
    wall = time.perf_counter()
    print("Compiling...", flush=True)
    _, _, accuracy, _ = train_step(params, opt_state, graphs)
    print(f"Compilation took {time.perf_counter() - wall:.1f}s")
    print(f"Initial accuracy = {100 * accuracy:.2f}%", flush=True)
    
    # Train
    wall = time.perf_counter()
    print("Training...", flush=True)
    with tqdm(range(steps)) as bar:
        for step in bar:
            params, opt_state, accuracy, preds = train_step(params, opt_state, graphs)
            bar.set_postfix(accuracy=f"{100 * accuracy:.2f}%")    
            if accuracy == 1.0:
                break
    print(f"Final accuracy = {100 * accuracy:.2f}%")
    print("Final prediction:", preds)

    # Check equivariance.
    print("Checking equivariance...")
    for key in range(10):
        key = jax.random.PRNGKey(key)
        alpha, beta, gamma = jax.random.uniform(key, (3,), minval=-jnp.pi, maxval=jnp.pi)
        
        rotated_nodes = e3nn.IrrepsArray("1o", graphs.nodes)
        rotated_nodes = rotated_nodes.transform_by_angles(alpha, beta, gamma)
        rotated_nodes = rotated_nodes.array
        rotated_graphs = graphs._replace(
            nodes=rotated_nodes
        )

        logits = model.apply(params, graphs)
        rotated_logits = model.apply(params, rotated_graphs)
        assert jnp.allclose(logits, rotated_logits, atol=1e-4), "Model is not equivariant."

    print("Model is equivariant.")


if __name__ == "__main__":
    train()