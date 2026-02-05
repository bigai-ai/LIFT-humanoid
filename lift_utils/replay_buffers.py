# Copyright 2023 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay buffers for Brax."""

import abc
from typing import Generic, Tuple, TypeVar
import flax
import jax
from jax import flatten_util
import jax.numpy as jnp
from lift_utils.types import PRNGKey

State = TypeVar('State')
Sample = TypeVar('Sample')


class ReplayBuffer(abc.ABC, Generic[State, Sample]):
    """Contains replay buffer methods."""

    @abc.abstractmethod
    def init(self, key: PRNGKey) -> State:
        """Init the replay buffer."""

    def insert(self, buffer_state: State, samples: Sample) -> State:
        """Insert data into the replay buffer."""
        self.check_can_insert(buffer_state, samples, 1)
        return self.insert_internal(buffer_state, samples)

    def sample(self, buffer_state: State) -> Tuple[State, Sample]:
        """Sample a batch of data."""
        self.check_can_sample(buffer_state, 1)
        return self.sample_internal(buffer_state)

    def check_can_insert(self, buffer_state: State, samples: Sample, shards: int):
        """Checks whether insert can be performed. Do not JIT this method."""
        pass

    def check_can_sample(self, buffer_state: State, shards: int):
        """Checks whether sampling can be performed. Do not JIT this method."""
        pass

    @abc.abstractmethod
    def size(self, buffer_state: State) -> int:
        """Total amount of elements that are sampleable."""

    @abc.abstractmethod
    def insert_internal(self, buffer_state: State, samples: Sample) -> State:
        """Insert data into the replay buffer."""

    @abc.abstractmethod
    def sample_internal(self, buffer_state: State) -> Tuple[State, Sample]:
        """Sample a batch of data."""


@flax.struct.dataclass
class ReplayBufferState:
    """Contains data related to a replay buffer."""

    data: jnp.ndarray
    insert_position: jnp.ndarray
    sample_position: jnp.ndarray
    key: PRNGKey


class QueueBase(ReplayBuffer[ReplayBufferState, Sample], Generic[Sample]):
    """Base class for limited-size FIFO reply buffers.

    Implements an `insert()` method which behaves like a limited-size queue.
    I.e. it adds samples to the end of the queue and, if necessary, removes the
    oldest samples form the queue in order to keep the maximum size within the
    specified limit.

    Derived classes must implement the `sample()` method.
    """

    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample: Sample,
        sample_batch_size: int,
    ):
        self._flatten_fn = jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0])

        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(
            dummy_data_sample
        )
        self._unflatten_fn = jax.vmap(self._unflatten_fn)
        data_size = len(dummy_flatten)

        self._data_shape = (max_replay_size, data_size)
        self._data_dtype = dummy_flatten.dtype
        self._sample_batch_size = sample_batch_size
        self._size = 0

    def init(self, key: PRNGKey) -> ReplayBufferState:
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_dtype),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def check_can_insert(self, buffer_state, samples, shards):
        """Checks whether insert operation can be performed."""
        assert isinstance(shards, int), 'This method should not be JITed.'
        insert_size = jax.tree_flatten(samples)[0][0].shape[0] // shards
        if self._data_shape[0] < insert_size:
            raise ValueError(
                'Trying to insert a batch of samples larger than the maximum replay'
                f' size. num_samples: {insert_size}, max replay size'
                f' {self._data_shape[0]}'
            )
        self._size = min(self._data_shape[0], self._size + insert_size)

    def insert_internal(
        self, buffer_state: ReplayBufferState, samples: Sample
    ) -> ReplayBufferState:
        """Insert data in the replay buffer.

        Args:
            buffer_state: Buffer state
            samples: Sample to insert with a leading batch size.

        Returns:
            New buffer state.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f'buffer_state.data.shape ({buffer_state.data.shape}) '
                f"doesn't match the expected value ({self._data_shape})"
            )

        update = self._flatten_fn(samples)
        data = buffer_state.data

        # If needed, roll the buffer to make sure there's enough space to fit
        # `update` after the current position.
        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(
            roll, lambda: jnp.roll(data, roll, axis=0), lambda: data
        )
        position = position + roll

        # Update the buffer and the control numbers.
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % (len(data) + 1)
        sample_position = jnp.maximum(0, buffer_state.sample_position + roll)

        return buffer_state.replace(
            data=data,
            insert_position=position,
            sample_position=sample_position,
        )

    def sample_internal(
        self, buffer_state: ReplayBufferState
    ) -> Tuple[ReplayBufferState, Sample]:
        raise NotImplementedError(f'{self.__class__}.sample() is not implemented.')

    def size(self, buffer_state: ReplayBufferState) -> int:
        return buffer_state.insert_position - buffer_state.sample_position  # pytype: disable=bad-return-type  # jax-ndarray


class UniformSamplingQueue(QueueBase[Sample], Generic[Sample]):
    """Implements an uniform sampling limited-size replay queue.

    * It behaves as a limited size queue (if buffer is full it removes the oldest
      elements when new one is inserted).
    * It supports batch insertion only (no single element)
    * It performs uniform random sampling with replacement of a batch of size
      `sample_batch_size`
    """

    def sample_internal(
        self, buffer_state: ReplayBufferState
    ) -> Tuple[ReplayBufferState, Sample]:
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f'Data shape expected by the replay buffer ({self._data_shape}) does '
                f'not match the shape of the buffer state ({buffer_state.data.shape})'
            )

        key, sample_key = jax.random.split(buffer_state.key)
        idx = jax.random.randint(
            sample_key,
            (self._sample_batch_size,),
            minval=buffer_state.sample_position,
            maxval=buffer_state.insert_position,
        )
        batch = jnp.take(buffer_state.data, idx, axis=0, mode='wrap')
        return buffer_state.replace(key=key), self._unflatten_fn(batch)

