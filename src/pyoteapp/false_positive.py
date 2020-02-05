import numpy as np
from numpy import random, convolve
from numba import prange, njit, int64, float64
# import time
import matplotlib
# import matplotlib.pyplot as plt
# from pyoteapp.autocorrtools import *
matplotlib.use('Qt5Agg')


@njit(float64[:](int64, float64), cache=True)
def noise_gen_jit(num_points: int, noise_sigma: float) -> np.ndarray:
    out = np.empty(num_points)
    for i in range(num_points):
        out[i] = random.normal(0.0, noise_sigma)
    return out


@njit(cache=True)
def simple_convolve(x, c):
    nx = len(x)
    nc = len(c)
    nout = nx - nc + 1
    out = np.zeros(nout)
    for i in range(nout):
        for j in range(nc):
            out[i] += x[i + j] * c[j]
    return out


@njit(cache=True)
def simple_convolve2(x, c):
    nx = len(x)
    nc = len(c)
    nout = nx - nc + 1
    out = np.zeros(nout)
    i = 0
    while i < nout:
        j = 0
        while j < nc:
            out[i] += x[i + j] * c[j]
            j += 1
        i += 1
    return out


def noise_gen_numpy(num_points: int, noise_sigma: float) -> np.ndarray:
    return random.normal(size=num_points, scale=noise_sigma)


# @jit(float64[:](int64, float64, float64[:]))
def simulated_observation(observation_size: int, noise_sigma: float, corr_array: np.ndarray) -> np.ndarray:
    """Returns a numpy array of floats representing an observation that contains only correlated noise.

    observation_size: number of points in the simulated observation
    noise_sigma:      standard deviation of observation noise
    corr_array:      correlation array needed to produce the correlated noise
    """
    return convolve(noise_gen_jit(observation_size, noise_sigma), corr_array, 'same')
    # noise_array = noise_gen_jit(observation_size+len(corr_array), noise_sigma)
    # out = np.empty(observation_size)
    # for i in prange(observation_size):
    #     out[i] = np.dot(corr_array, noise_array[i:i+len(corr_array)])
    #
    # return out


# noinspection PyPep8Naming
@njit(cache=True)
def max_drop_in_simulated_observation(
        event_duration: int,
        observation_size: int,
        noise_sigma: float,
        corr_array: np.ndarray) -> float:
    """Returns the maximum drop from a simulated observation.

    An observation consisting only of correlated gaussian noise is
    created and exhaustively searched for an 'event' of a specified size.
    The biggest 'drop' (B - A) found is returned.  This drop will always
    be positive (>= 0.0)

    event_duration:   size/duration of event being simulated
    observation_size: number of points in the simulated observation
    noise_sigma:      standard deviation of observation noise
    corr_array:       correlation array needed to produce the correlated noise
    """

    # obs = simulated_observation(observation_size, noise_sigma, corr_array)
    obs = noise_gen_jit(observation_size, noise_sigma)
    obs = simple_convolve(obs, corr_array)

    obs_size = len(obs)

    numA = event_duration
    numB = obs_size - numA

    # Start with the trial 'event' position at the extreme left of the observation
    cumA = np.sum(obs[0:numA])
    cumB = np.sum(obs[numA:])

    A = cumA / numA
    B = cumB / numB

    drop = B - A

    best_drop_so_far = drop if drop >= 0.0 else 0.0

    # Calculate new values for B and A iteratively and conditionally update best_drop_so_far
    for i in range(1, numB):
        cumA -= obs[i]
        cumB += obs[i]

        cumA += obs[i + event_duration]
        cumB -= obs[i + event_duration]

        A = cumA / numA
        B = cumB / numB

        drop = B - A
        if drop > best_drop_so_far:
            best_drop_so_far = drop

    return best_drop_so_far


@njit(parallel=True)
def compute_drops(event_duration: int,
                  observation_size: int,
                  noise_sigma: float,
                  corr_array: np.ndarray,
                  num_trials: int) -> np.ndarray:
    """Returns the maximum drops from num_trials simulated observations.

    event_duration:   size/duration of event being simulated
    observation_size: number of points in the simulated observation
    noise_sigma:      standard deviation of observation noise
    corr_array:       correlation array needed to produce the correlated noise
    num_trials:       number of simulated observations
    """

    drops: np.ndarray = np.empty(num_trials)

    for i in prange(num_trials):
        drops[i] = max_drop_in_simulated_observation(
            event_duration=event_duration, observation_size=observation_size,
            noise_sigma=noise_sigma, corr_array=corr_array)

    return drops


# def excercise():
#     noise_gen_jit(1, 1.0)
#     noise_gen_jit(1, 1.0)
#
#     start_time = time.time()
#     _ = noise_gen_jit(4_000, 1.0)
#     stop_time = time.time()
#     print(f'noise_gen_jit time: {stop_time - start_time:.6f} seconds')
#
#     start_time = time.time()
#     _ = noise_gen_numpy(4_000, 1.0)
#     stop_time = time.time()
#     print(f'noise_gen_numpy time: {stop_time - start_time:.6f} seconds')
#
#     # Warm-up calls to allow Numba compiler to do its work
#     print(f'\nDoing Numba warm-up calls...')
#     simulated_observation(100, 1.0, np.ones(10))
#     simulated_observation(100, 1.0, np.ones(10))
#
#     max_drop_in_simulated_observation(10, 100, 1.0, np.ones(10))
#     max_drop_in_simulated_observation(10, 100, 1.0, np.ones(10))
#
#     compute_drops(10, 100, 1.0, np.ones(10), 1)
#     compute_drops(10, 100, 1.0, np.ones(10), 1)
#     print(f'...finished warm-up calls.\n')
#     # End warm-up calls
#
#     dur = 100
#     n_obs = 4000
#     sigma = 2.0
#     # corr = np.ones(10)
#     corr = np.array([0.11653067, 0.29474461, 0.40390159, 0.85814318])
#     n_trials = 10000
#
#     start_time = time.time()
#     drop_array = compute_drops(event_duration=dur, observation_size=n_obs,
#                                noise_sigma=sigma, corr_array=corr, num_trials=n_trials)
#     stop_time = time.time()
#     print(f'{n_trials} drops computed in {stop_time - start_time:.6f} seconds')
#
#     # %%
#     plt.plot(np.sort(drop_array))
#     plt.show()
#
#     # %%
#     hist_values, bin_edges = np.histogram(drop_array, 50)
#     # print(hist_values)
#     # print(bin_edges)
#     plt.hist(drop_array, bins=50)
#     max_drop = np.max(drop_array)
#     plt.vlines(max_drop, 0, max(hist_values), colors='r')
#     plt.show()
#
#     noise_gen = CorrelatedNoiseGenerator(acfcoeffs=[1.0, 0.5, 0.3, 0.1])
#     test_noise = noise_gen.corr_normal(num_values=100000)
#     print(f'sigma: {np.std(test_noise)}')
#     print(autocorr(test_noise, lastlag=6))
#     print(noise_gen.final_coeffs)
#

# if __name__ == "__main__":
#     excercise()
