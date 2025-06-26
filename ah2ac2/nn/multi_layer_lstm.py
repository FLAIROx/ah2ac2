import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from typing import List


class LstmCellWithHiddenStateReset(nn.OptimizedLSTMCell):
    """
    LSTM Cell that handles resetting the carry when the resets are passed alongside inputs.
    """

    def __call__(self, carry, x):
        # NOTE: We handle two cases here:
        #  - When resets is passed alongside inputs, we reset the carry if needed.
        #  - When x contains just inputs, we simply call the LSTMCell.
        if isinstance(x, tuple):
            x, resets = x
            init_rnn_state = self.initialize_carry(jax.random.PRNGKey(0), (x.shape[0], x.shape[1]))
            carry = jax.tree_map(
                lambda init, old: jnp.where(resets[:, np.newaxis], init, old),
                init_rnn_state,
                carry,
            )

        out = super().__call__(carry, x)
        return out


class MultiLayerLstm(nn.Module):
    preprocessing_features: List[int]
    lstm_features: List[int]
    postprocessing_features: List[int]
    action_dim: int
    dropout_rate: float
    activation_fn_name: str

    def setup(self):
        self.lstm = nn.scan(
            LstmCellWithHiddenStateReset,
            variable_broadcast="params",
            in_axes=1,
            out_axes=1,
            split_rngs={"params": False},
        )
        if self.activation_fn_name == "relu":
            self.act_fn = nn.relu
        elif self.activation_fn_name == "gelu":
            self.act_fn = nn.gelu
        else:  # Default to relu
            self.act_fn = nn.relu

    @nn.compact
    def __call__(self, carry, x, training: bool):
        # Preprocess features:
        for feature_size in self.preprocessing_features:
            x = nn.Dense(feature_size)(x)
            x = nn.LayerNorm()(x)
            x = self.act_fn(x)

        # Run LSTM layers and get new hidden states:
        new_cs, new_hs = [], []
        for i, feature_size in enumerate(self.lstm_features):
            layer = self.lstm(feature_size)

            layer_carry = jax.tree_map(lambda c: c[i], carry)

            layer_carry, x = layer(layer_carry, x)
            new_c, new_h = layer_carry
            new_cs.append(new_c)
            new_hs.append(new_h)

        # Postprocess features:
        for feature_size in self.postprocessing_features:
            x = nn.Dense(feature_size)(x)
            x = self.act_fn(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        x = nn.Dense(self.action_dim)(x)
        new_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_carry, x

    @nn.nowrap
    def initialize_carry(self, batch_size):
        carry_hs = []
        carry_cs = []

        in_features = self.preprocessing_features[-1]
        for feature_size in self.lstm_features:
            layer = nn.OptimizedLSTMCell(feature_size)

            c, h = layer.initialize_carry(jax.random.PRNGKey(0), (batch_size, in_features))
            carry_cs.append(c)
            carry_hs.append(h)

            in_features = feature_size

        return jnp.stack(carry_cs), jnp.stack(carry_hs)
