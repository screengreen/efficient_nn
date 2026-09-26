import numpy as np


def output_size(S, kernel_size, stride=1, padding=0):
    return (S + 2 * padding - kernel_size) // stride + 1


def conv2d_flops(S, B, in_channels, out_channels, kernel_size, stride=1, padding=0):
    S_out = output_size(S, kernel_size, stride, padding)

    return (
        2
        * B
        * kernel_size**2
        * in_channels
        * S_out**2
        * out_channels
    )


def maxpool2d_flops(S, B, channels, kernel_size, stride, padding=0):
    S_out = output_size(S, kernel_size, stride, padding)

    return B * S_out**2 * channels * (kernel_size**2 - 1)


def relu_flops(S, B, channels):
    return B * S**2 * channels


def global_avg_pool_flops(S, B, channels):
    return B * channels * S**2


def linear_flops(B, in_features, out_features):
    return 2 * B * in_features * out_features


def flops(image_size, batch):
    S = np.asarray(image_size)
    B = np.asarray(batch)

    total = 0

    total += conv2d_flops(S, B, 3, 32, 7, 2, 3)
    S = output_size(S, 7, 2, 3)
    total += relu_flops(S, B, 32)

    total += maxpool2d_flops(S, B, 32, 3, 2, 1)
    S = output_size(S, 3, 2, 1)

    total += conv2d_flops(S, B, 32, 64, 5, 1, 2)
    total += relu_flops(S, B, 64)

    total += conv2d_flops(S, B, 64, 128, 3, 2, 1)
    S = output_size(S, 3, 2, 1)
    total += relu_flops(S, B, 128)

    total += conv2d_flops(S, B, 128, 256, 1)
    total += relu_flops(S, B, 256)

    total += conv2d_flops(S, B, 256, 256, 3, 2, 1)
    S = output_size(S, 3, 2, 1)
    total += relu_flops(S, B, 256)

    total += conv2d_flops(S, B, 256, 512, 1)
    total += relu_flops(S, B, 512)

    total += global_avg_pool_flops(S, B, 512)

    total += linear_flops(B, 512, 256)
    total += B * 256

    total += linear_flops(B, 256, 100)

    return total


def tensor_memory(S, B, channels, bytes_per_element=4):
    return B * channels * S**2 * bytes_per_element


def linear_tensor_memory(B, features, bytes_per_element=4):
    return B * features * bytes_per_element


def conv2d_parameters_memory(
    in_channels,
    out_channels,
    kernel_size,
    bytes_per_element=4,
):
    return (
        in_channels
        * out_channels
        * kernel_size**2
        * bytes_per_element
    )


def linear_parameters_memory(
    in_features,
    out_features,
    bytes_per_element=4,
):
    return (
        (in_features * out_features + out_features)
        * bytes_per_element
    )


def memory(image_size, batch):
    S = np.asarray(image_size)
    B = np.asarray(batch)

    parameters = 0

    parameters += conv2d_parameters_memory(3, 32, 7)
    parameters += conv2d_parameters_memory(32, 64, 5)
    parameters += conv2d_parameters_memory(64, 128, 3)
    parameters += conv2d_parameters_memory(128, 256, 1)
    parameters += conv2d_parameters_memory(256, 256, 3)
    parameters += conv2d_parameters_memory(256, 512, 1)

    parameters += linear_parameters_memory(512, 256)
    parameters += linear_parameters_memory(256, 100)

    peaks = []

    S_out = output_size(S, 7, 2, 3)
    peaks.append(
        tensor_memory(S, B, 3)
        + tensor_memory(S_out, B, 32)
    )
    S = S_out

    S_out = output_size(S, 3, 2, 1)
    peaks.append(
        tensor_memory(S, B, 32)
        + tensor_memory(S_out, B, 32)
    )
    S = S_out

    S_out = output_size(S, 5, 1, 2)
    peaks.append(
        tensor_memory(S, B, 32)
        + tensor_memory(S_out, B, 64)
    )
    S = S_out

    S_out = output_size(S, 3, 2, 1)
    peaks.append(
        tensor_memory(S, B, 64)
        + tensor_memory(S_out, B, 128)
    )
    S = S_out

    S_out = output_size(S, 1)
    peaks.append(
        tensor_memory(S, B, 128)
        + tensor_memory(S_out, B, 256)
    )
    S = S_out

    S_out = output_size(S, 3, 2, 1)
    peaks.append(
        tensor_memory(S, B, 256)
        + tensor_memory(S_out, B, 256)
    )
    S = S_out

    S_out = output_size(S, 1)
    peaks.append(
        tensor_memory(S, B, 256)
        + tensor_memory(S_out, B, 512)
    )
    S = S_out

    peaks.append(
        tensor_memory(S, B, 512)
        + linear_tensor_memory(B, 512)
    )

    peaks.append(
        linear_tensor_memory(B, 512)
        + linear_tensor_memory(B, 256)
    )

    peaks.append(
        linear_tensor_memory(B, 256)
        + linear_tensor_memory(B, 100)
    )

    activations_peak = np.maximum.reduce(peaks)

    return parameters + activations_peak




def bytes_moved(image_size, batch):
    S = np.asarray(image_size)
    B = np.asarray(batch)

    total = 0

    S_out = output_size(S, 7, 2, 3)
    total += (
        tensor_memory(S, B, 3)
        + conv2d_parameters_memory(3, 32, 7)
        + tensor_memory(S_out, B, 32)
    )
    S = S_out

    S_out = output_size(S, 3, 2, 1)
    total += (
        tensor_memory(S, B, 32)
        + tensor_memory(S_out, B, 32)
    )
    S = S_out

    S_out = output_size(S, 5, 1, 2)
    total += (
        tensor_memory(S, B, 32)
        + conv2d_parameters_memory(32, 64, 5)
        + tensor_memory(S_out, B, 64)
    )
    S = S_out

    S_out = output_size(S, 3, 2, 1)
    total += (
        tensor_memory(S, B, 64)
        + conv2d_parameters_memory(64, 128, 3)
        + tensor_memory(S_out, B, 128)
    )
    S = S_out

    S_out = output_size(S, 1)
    total += (
        tensor_memory(S, B, 128)
        + conv2d_parameters_memory(128, 256, 1)
        + tensor_memory(S_out, B, 256)
    )
    S = S_out

    S_out = output_size(S, 3, 2, 1)
    total += (
        tensor_memory(S, B, 256)
        + conv2d_parameters_memory(256, 256, 3)
        + tensor_memory(S_out, B, 256)
    )
    S = S_out

    S_out = output_size(S, 1)
    total += (
        tensor_memory(S, B, 256)
        + conv2d_parameters_memory(256, 512, 1)
        + tensor_memory(S_out, B, 512)
    )
    S = S_out

    total += (
        tensor_memory(S, B, 512)
        + linear_tensor_memory(B, 512)
    )

    total += (
        linear_tensor_memory(B, 512)
        + linear_parameters_memory(512, 256)
        + linear_tensor_memory(B, 256)
    )

    total += (
        linear_tensor_memory(B, 256)
        + linear_parameters_memory(256, 100)
        + linear_tensor_memory(B, 100)
    )

    return total



def latency(image_size, batch, theta):
    """theta = [launch floor in seconds, seconds per GFLOP, slowdown per GB].

    Time is at least one launch. Above that it is the FLOPs, run at a peak
    that falls as more bytes move: a larger working set is less cache-friendly.
    """
    f = flops(image_size, batch) / 1e9
    m = bytes_moved(image_size, batch) / 1e9

    return np.maximum(theta[0], theta[1] * f + (1-theta[2]) * m)


def energy(image_size, batch, theta):
    """theta = [floor in joules, joules per GFLOP, slowdown per GB].

    A launch costs a fixed energy. Above that each GFLOP costs more as more
    bytes move: the board draws about 67 W for the whole run, so energy
    follows the time spent.
    """
    f = flops(image_size, batch) / 1e9
    m = bytes_moved(image_size, batch) / 1e9

    return np.maximum(theta[0], theta[1] * f * (1 + theta[2] * m))