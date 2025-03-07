# coding=utf-8
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The standard DQN replay memory.

This implementation is an out-of-graph replay memory + in-graph wrapper. It
supports vanilla n-step updates of the form typically found in the literature,
i.e. where rewards are accumulated for n steps and the intermediate trajectory
is not exposed to the agent. This does not allow, for example, performing
off-policy corrections.
"""
import collections
import math

import numpy as np

# Defines a type describing part of the tuple returned by the replay
# memory. Each element of the tuple is a tensor of shape [batch, ...] where
# ... is defined the 'shape' field of ReplayElement. The tensor type is
# given by the 'type' field. The 'name' field is for convenience and ease of
# debugging.
ReplayElement = collections.namedtuple("shape_type", ["name", "shape", "type"])


def modulo_range(start, length, modulo):
    for i in range(length):
        yield int((start + i) % modulo)


def invalid_range(cursor, replay_capacity, stack_size, update_horizon):
    """Returns a array with the indices of cursor-related invalid transitions.

    There are update_horizon + stack_size invalid indices:
      - The update_horizon indices before the cursor, because we do not have a
        valid N-step transition (including the next state).
      - The stack_size indices on or immediately after the cursor.
    If N = update_horizon, K = stack_size, and the cursor is at c, invalid
    indices are:
      c - N, c - N + 1, ..., c, c + 1, ..., c + K - 1.

    It handles special cases in a circular buffer in the beginning and the end.

    Args:
      cursor: int, the position of the cursor.
      replay_capacity: int, the size of the replay memory.
      stack_size: int, the size of the stacks returned by the replay memory.
      update_horizon: int, the agent's update horizon.
    Returns:
      np.array of size stack_size with the invalid indices.
    """
    assert cursor < replay_capacity
    return np.array([(cursor - update_horizon + i) % replay_capacity for i in range(stack_size + update_horizon)])


class ReplayBuffer(object):
    """A simple out-of-graph Replay Buffer.

    Stores transitions, state, action, reward, next_state, terminal (and any
    extra contents specified) in a circular buffer and provides a uniform
    transition sampling function.

    When the states consist of stacks of observations storing the states is
    inefficient. This class writes observations and constructs the stacked states
    at sample time.

    Attributes:
      add_count: int, counter of how many transitions have been added (including
        the blank ones at the beginning of an episode).
      invalid_range: np.array, an array with the indices of cursor-related invalid
        transitions
      episode_end_indices: set[int], a set of indices corresponding to the
        end of an episode.
    """

    def __init__(
        self,
        observation_shape,
        replay_capacity,
        batch_size,
        update_horizon,
        gamma,
        clipping=lambda x: x,
        stack_size=1,
        max_sample_attempts=1000,
        extra_storage_types=None,
        observation_dtype=np.float32,
        terminal_dtype=bool,
        action_shape=(),
        action_dtype=np.int32,
        reward_shape=(),
        reward_dtype=np.float32,
    ):
        """Initializes OutOfGraphReplayBuffer.

        Args:
          observation_shape: tuple of ints.
          stack_size: int, number of frames to use in state stack.
          replay_capacity: int, number of transitions to keep in memory.
          batch_size: int.
          update_horizon: int, length of update ('n' in n-step update).
          gamma: float, the discount factor.
          max_sample_attempts: int, the maximum number of attempts allowed to
            get a sample.
          extra_storage_types: list of ReplayElements defining the type of the extra
            contents that will be stored and returned by sample_transition_batch.
          observation_dtype: np.dtype, type of the observations. Defaults to
            np.uint8 for Atari 2600.
          terminal_dtype: np.dtype, type of the terminals. Defaults to np.uint8 for
            Atari 2600.
          action_shape: tuple of ints, the shape for the action vector. Empty tuple
            means the action is a scalar.
          action_dtype: np.dtype, type of elements in the action.
          reward_shape: tuple of ints, the shape of the reward vector. Empty tuple
            means the reward is a scalar.
          reward_dtype: np.dtype, type of elements in the reward.

        Raises:
          ValueError: If replay_capacity is too small to hold at least one
            transition.
        """
        assert isinstance(observation_shape, tuple)
        if replay_capacity < update_horizon + stack_size:
            raise ValueError("There is not enough capacity to cover " "update_horizon and stack_size.")

        self._action_shape = action_shape
        self._action_dtype = action_dtype
        self._reward_shape = reward_shape
        self._reward_dtype = reward_dtype
        self._observation_shape = observation_shape
        self._stack_size = stack_size
        self._state_shape = self._observation_shape + (self._stack_size,)
        self._replay_capacity = replay_capacity
        self._batch_size = batch_size
        self._update_horizon = update_horizon
        self._gamma = gamma
        self._clipping = clipping
        self._observation_dtype = observation_dtype
        self._terminal_dtype = terminal_dtype
        self._max_sample_attempts = max_sample_attempts
        if extra_storage_types:
            self._extra_storage_types = extra_storage_types
        else:
            self._extra_storage_types = []
        self._create_storage()
        self.add_count = np.array(0)
        self.invalid_range = np.zeros((self._stack_size))
        # When the horizon is > 1, we compute the sum of discounted rewards as a dot
        # product using the precomputed vector <gamma^0, gamma^1, ..., gamma^{n-1}>.
        self._cumulative_discount_vector = np.array(
            [math.pow(self._gamma, n) for n in range(update_horizon)], dtype=np.float32
        )
        self._next_experience_is_episode_start = True
        self.episode_end_indices = set()

    @classmethod
    def reset(cls, rb):
        return cls(
            rb._observation_shape,
            rb._replay_capacity,
            rb._batch_size,
            rb._update_horizon,
            rb._gamma,
            rb._clipping,
            rb._stack_size,
            rb._max_sample_attempts,
            rb._extra_storage_types,
            rb._observation_dtype,
            rb._terminal_dtype,
            rb._action_shape,
            rb._action_dtype,
            rb._reward_shape,
            rb._reward_dtype,
        )

    def _create_storage(self):
        """Creates the numpy arrays used to store transitions."""
        self._store = {}
        for storage_element in self.get_storage_signature():
            array_shape = [self._replay_capacity] + list(storage_element.shape)
            self._store[storage_element.name] = np.empty(array_shape, dtype=storage_element.type)

    def get_add_args_signature(self):
        """The signature of the add function.

        Note - Derived classes may return a different signature.

        Returns:
          list of ReplayElements defining the type of the argument signature needed
            by the add function.
        """
        return self.get_storage_signature()

    def get_storage_signature(self):
        """Returns a default list of elements to be stored in this replay memory.

        Note - Derived classes may return a different signature.

        Returns:
          list of ReplayElements defining the type of the contents stored.
        """
        storage_elements = [
            ReplayElement("observation", self._observation_shape, self._observation_dtype),
            ReplayElement("action", self._action_shape, self._action_dtype),
            ReplayElement("reward", self._reward_shape, self._reward_dtype),
            ReplayElement("terminal", (), self._terminal_dtype),
        ]

        for extra_replay_element in self._extra_storage_types:
            storage_elements.append(extra_replay_element)
        return storage_elements

    def _add_zero_transition(self):
        """Adds a padding transition filled with zeros (Used in episode beginnings)."""
        zero_transition = []
        for element_type in self.get_add_args_signature():
            zero_transition.append(np.zeros(element_type.shape, dtype=element_type.type))
        self.episode_end_indices.discard(self.cursor())  # If present
        self._add(*zero_transition)

    def add(self, observation, action, reward, terminal, *args, priority=None, episode_end=False):
        """Adds a transition to the replay memory.

        This function checks the types and handles the padding at the beginning of
        an episode. Then it calls the _add function.

        Since the next_observation in the transition will be the observation added
        next there is no need to pass it.

        If the replay memory is at capacity the oldest transition will be discarded.

        Args:
          observation: np.array with shape observation_shape.
          action: int, the action in the transition.
          reward: float, the reward received in the transition.
          terminal: np.dtype, acts as a boolean indicating whether the transition
                    was terminal (1) or not (0).
          *args: extra contents with shapes and dtypes according to
            extra_storage_types.
          priority: float, unused in the circular replay buffer, but may be used
            in child classes like PrioritizedReplayBuffer.
          episode_end: bool, whether this experience is the last experience in
            the episode. This is useful for tasks that terminate due to time-out,
            but do not end on a terminal state. Overloading 'terminal' may not
            be sufficient in this case, since 'terminal' is passed to the agent
            for training. 'episode_end' allows the replay buffer to determine
            episode boundaries without passing that information to the agent.
        """
        if priority is not None:
            args = args + (priority,)

        self._check_add_types(observation, action, reward, terminal, *args)
        if self._next_experience_is_episode_start:
            for _ in range(self._stack_size - 1):
                # Child classes can rely on the padding transitions being filled with
                # zeros. This is useful when there is a priority argument.
                self._add_zero_transition()
            self._next_experience_is_episode_start = False

        if episode_end or terminal:
            self.episode_end_indices.add(self.cursor())
            self._next_experience_is_episode_start = True
        else:
            self.episode_end_indices.discard(self.cursor())  # If present

        self._add(observation, action, self._clipping(reward), terminal, *args)

    def _add(self, *args):
        """Internal add method to add to the storage arrays.

        Args:
          *args: All the elements in a transition.
        """
        self._check_args_length(*args)
        transition = {e.name: args[idx] for idx, e in enumerate(self.get_add_args_signature())}
        self._add_transition(transition)

    def _add_transition(self, transition):
        """Internal add method to add transition dictionary to storage arrays.

        Args:
          transition: The dictionary of names and values of the transition
                      to add to the storage.
        """
        cursor = self.cursor()
        for arg_name in transition:
            self._store[arg_name][cursor] = transition[arg_name]

        self.add_count += 1
        self.invalid_range = invalid_range(self.cursor(), self._replay_capacity, self._stack_size, self._update_horizon)

    def _check_args_length(self, *args):
        """Check if args passed to the add method have the same length as storage.

        Args:
          *args: Args for elements used in storage.

        Raises:
          ValueError: If args have wrong length.
        """
        if len(args) != len(self.get_add_args_signature()):
            raise ValueError(
                "Add expects {} elements, received {}".format(len(self.get_add_args_signature()), len(args))
            )

    def _check_add_types(self, *args):
        """Checks if args passed to the add method match those of the storage.

        Args:
          *args: Args whose types need to be validated.

        Raises:
          ValueError: If args have wrong shape or dtype.
        """
        self._check_args_length(*args)
        for arg_element, store_element in zip(args, self.get_add_args_signature()):
            if isinstance(arg_element, np.ndarray):
                arg_shape = arg_element.shape
            elif isinstance(arg_element, tuple) or isinstance(arg_element, list):
                # TODO(b/80536437). This is not efficient when arg_element is a list.
                arg_shape = np.array(arg_element).shape
            else:
                # Assume it is scalar.
                arg_shape = tuple()
            store_element_shape = tuple(store_element.shape)
            if arg_shape != store_element_shape:
                raise ValueError("arg has shape {}, expected {}".format(arg_shape, store_element_shape))

    def is_empty(self):
        """Is the Replay Buffer empty?"""
        return self.add_count == 0

    def is_full(self):
        """Is the Replay Buffer full?"""
        return self.add_count >= self._replay_capacity

    def cursor(self):
        """Index to the location where the next transition will be written."""
        return int(self.add_count % self._replay_capacity)

    def get_range(self, array, start_index, end_index):
        """Returns the range of array at the index handling wraparound if necessary.

        Args:
          array: np.array, the array to get the stack from.
          start_index: int, index to the start of the range to be returned. Range
            will wraparound if start_index is smaller than 0.
          end_index: int, exclusive end index. Range will wraparound if end_index
            exceeds replay_capacity.

        Returns:
          np.array, with shape [end_index - start_index, array.shape[1:]].
        """
        assert end_index > start_index, "end_index must be larger than start_index"
        assert end_index >= 0
        assert start_index < self._replay_capacity
        if not self.is_full():
            assert end_index <= self.cursor(), "Index {} has not been added.".format(start_index)

        # Fast slice read when there is no wraparound.
        if start_index % self._replay_capacity < end_index % self._replay_capacity:
            return_array = array[start_index:end_index, ...]
        # Slow list read.
        else:
            indices = [(start_index + i) % self._replay_capacity for i in range(end_index - start_index)]
            return_array = array[indices, ...]
        return return_array

    def get_observation_stack(self, index):
        return self._get_element_stack(index, "observation")

    def _get_element_stack(self, index, element_name):
        state = self.get_range(self._store[element_name], index - self._stack_size + 1, index + 1)
        # The stacking axis is 0 but the agent expects as the last axis.
        return np.moveaxis(state, 0, -1)

    def get_terminal_stack(self, index):
        return self.get_range(self._store["terminal"], index - self._stack_size + 1, index + 1)

    def is_valid_transition(self, index):
        """Checks if the index contains a valid transition.

        Checks for collisions with the end of episodes and the current position
        of the cursor.

        Args:
          index: int, the index to the state in the transition.

        Returns:
          Is the index valid: Boolean.

        """
        # Check the index is in the valid range
        if index < 0 or index >= self._replay_capacity:
            return False
        if not self.is_full():
            # The indices and next_indices must be smaller than the cursor.
            if index >= self.cursor() - self._update_horizon:
                return False
            # The first few indices contain the padding states of the first episode.
            if index < self._stack_size - 1:
                return False

        # Skip transitions that straddle the cursor.
        if index in set(self.invalid_range):
            return False

        # If there are terminal flags in any other frame other than the last one
        # the stack is not valid, so don't sample it.
        if self.get_terminal_stack(index)[:-1].any():
            return False

        # If the episode ends before the update horizon, without a terminal signal,
        # it is invalid.
        for i in modulo_range(index, self._update_horizon, self._replay_capacity):
            if i in self.episode_end_indices and not self._store["terminal"][i]:
                return False

        return True

    def _create_batch_arrays(self, batch_size):
        """Create a tuple of arrays with the type of get_transition_elements.

        When using the WrappedReplayBuffer with staging enabled it is important to
        create new arrays every sample because StaginArea keeps a pointer to the
        returned arrays.

        Args:
          batch_size: (int) number of transitions returned. If None the default
            batch_size will be used.

        Returns:
          Tuple of np.arrays with the shape and type of get_transition_elements.
        """
        transition_elements = self.get_transition_elements(batch_size)
        batch_arrays = []
        for element in transition_elements:
            batch_arrays.append(np.empty(element.shape, dtype=element.type))
        return tuple(batch_arrays)

    def sample_index_batch(self, batch_size):
        """Returns a batch of valid indices sampled uniformly.

        Args:
          batch_size: int, number of indices returned.

        Returns:
          list of ints, a batch of valid indices sampled uniformly.

        Raises:
          RuntimeError: If the batch was not constructed after maximum number of
            tries.
        """
        if self.is_full():
            # add_count >= self._replay_capacity > self._stack_size
            min_id = self.cursor() - self._replay_capacity + self._stack_size - 1
            max_id = self.cursor() - self._update_horizon
        else:
            # add_count < self._replay_capacity
            min_id = self._stack_size - 1
            max_id = self.cursor() - self._update_horizon
            if max_id <= min_id:
                raise RuntimeError(
                    "Cannot sample a batch with fewer than stack size "
                    "({}) + update_horizon ({}) transitions.".format(self._stack_size, self._update_horizon)
                )

        indices = []
        attempt_count = 0
        while len(indices) < batch_size and attempt_count < self._max_sample_attempts:
            index = np.random.randint(min_id, max_id) % self._replay_capacity
            if self.is_valid_transition(int(index)):
                indices.append(index)
            else:
                attempt_count += 1
        if len(indices) != batch_size:
            raise RuntimeError(
                "Max sample attempts: Tried {} times but only sampled {}"
                " valid indices. Batch size is {}".format(self._max_sample_attempts, len(indices), batch_size)
            )

        return indices

    def sample_transition_batch(self, batch_size=None, indices=None):
        """Returns a batch of transitions (including any extra contents).

        If get_transition_elements has been overridden and defines elements not
        stored in self._store, an empty array will be returned and it will be
        left to the child class to fill it. For example, for the child class
        OutOfGraphPrioritizedReplayBuffer, the contents of the
        sampling_probabilities are stored separately in a sum tree.

        When the transition is terminal next_state_batch has undefined contents.

        NOTE: This transition contains the indices of the sampled elements. These
        are only valid during the call to sample_transition_batch, i.e. they may
        be used by subclasses of this replay buffer but may point to different data
        as soon as sampling is done.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
          indices: None or list of ints, the indices of every transition in the
            batch. If None, sample the indices uniformly.

        Returns:
          transition_batch: tuple of np.arrays with the shape and type as in
            get_transition_elements().

        Raises:
          ValueError: If an element to be sampled is missing from the replay buffer.
        """
        if batch_size is None:
            batch_size = self._batch_size
        if indices is None:
            indices = self.sample_index_batch(batch_size)
        assert len(indices) == batch_size

        transition_elements = self.get_transition_elements(batch_size)
        batch_arrays = self._create_batch_arrays(batch_size)
        for batch_element, state_index in enumerate(indices):
            trajectory_indices = [(state_index + j) % self._replay_capacity for j in range(self._update_horizon)]
            trajectory_terminals = self._store["terminal"][trajectory_indices]
            is_terminal_transition = trajectory_terminals.any()
            if not is_terminal_transition:
                trajectory_length = self._update_horizon
            else:
                # np.argmax of a bool array returns the index of the first True.
                trajectory_length = np.argmax(trajectory_terminals.astype(bool), 0) + 1
            next_state_index = state_index + trajectory_length
            trajectory_discount_vector = self._cumulative_discount_vector[:trajectory_length]
            trajectory_rewards = self.get_range(self._store["reward"], state_index, next_state_index)

            # Fill the contents of each array in the sampled batch.
            assert len(transition_elements) == len(batch_arrays)
            for element_array, element in zip(batch_arrays, transition_elements):
                if element.name == "state":
                    element_array[batch_element] = self.get_observation_stack(state_index)
                elif element.name == "reward":
                    # compute the discounted sum of rewards in the trajectory.
                    element_array[batch_element] = np.sum(trajectory_discount_vector * trajectory_rewards, axis=0)
                elif element.name == "next_state":
                    element_array[batch_element] = self.get_observation_stack(
                        (next_state_index) % self._replay_capacity
                    )
                elif element.name in ("next_action", "next_reward"):
                    element_array[batch_element] = self._store[element.name.lstrip("next_")][
                        (next_state_index) % self._replay_capacity
                    ]
                elif element.name == "terminal":
                    element_array[batch_element] = is_terminal_transition
                elif element.name == "indices":
                    element_array[batch_element] = state_index
                elif element.name in self._store.keys():
                    element_array[batch_element] = self._store[element.name][state_index]
                # We assume the other elements are filled in by the subclass.

        return batch_arrays

    def get_transition_elements(self, batch_size=None):
        """Returns a 'type signature' for sample_transition_batch.

        Args:
          batch_size: int, number of transitions returned. If None, the default
            batch_size will be used.
        Returns:
          signature: A namedtuple describing the method's return type signature.
        """
        batch_size = self._batch_size if batch_size is None else batch_size

        transition_elements = [
            ReplayElement("state", (batch_size,) + self._state_shape, self._observation_dtype),
            ReplayElement("action", (batch_size,) + self._action_shape, self._action_dtype),
            ReplayElement("reward", (batch_size,) + self._reward_shape, self._reward_dtype),
            ReplayElement("next_state", (batch_size,) + self._state_shape, self._observation_dtype),
            ReplayElement("next_action", (batch_size,) + self._action_shape, self._action_dtype),
            ReplayElement("next_reward", (batch_size,) + self._reward_shape, self._reward_dtype),
            ReplayElement("terminal", (batch_size,), self._terminal_dtype),
            ReplayElement("indices", (batch_size,), np.int32),
        ]
        for element in self._extra_storage_types:
            transition_elements.append(ReplayElement(element.name, (batch_size,) + tuple(element.shape), element.type))
        return transition_elements

    def get_all_valid_samples(self):
        all_valid_indices = [idx for idx in range(self._replay_capacity) if self.is_valid_transition(idx)]

        return self.sample_transition_batch(batch_size=len(all_valid_indices), indices=all_valid_indices)
