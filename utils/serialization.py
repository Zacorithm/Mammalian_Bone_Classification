"""Pure-PyTorch space-filling-curve codecs (PointMamba, PT-V3): Hilbert (Skilling) and
Morton/z-order, plus batch-major composition so sorted sequences never interleave samples:
code = batch_id << (3 * depth) | curve_code(grid_coord). Grid coords must be non-negative ints.
"""

import math

import torch


ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")


# ---------------------------------------------------------------------------
# Hilbert curve (Skilling's algorithm)
# ---------------------------------------------------------------------------

def right_shift(binary, k=1, axis=-1):
    """Right shift an array of binary values (as bits along `axis`)."""
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )
    return shifted


def gray2binary(gray, axis=-1):
    """Convert an array of Gray codes back into binary values."""
    shift = 2 ** (int(math.ceil(math.log2(gray.shape[axis]))) - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift, axis=axis))
        shift = shift // 2
    return gray


def hilbert_encode_(locs, num_dims=3, num_bits=16):
    """locs (N, num_dims) non-negative ints < 2**num_bits -> (N,) int64 Hilbert codes."""
    orig_shape = locs.shape
    bitpack_mask = 1 << torch.arange(0, 8, device=locs.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    if orig_shape[-1] != num_dims:
        raise ValueError(
            f"locs last dim ({orig_shape[-1]}) must equal num_dims ({num_dims})"
        )
    if num_dims * num_bits > 63:
        raise ValueError(
            f"num_dims * num_bits = {num_dims * num_bits} exceeds the int64 budget (63)"
        )

    # int64 -> 8 bytes (little-endian) -> flip to big-endian byte order
    locs_uint8 = locs.long().contiguous().view(torch.uint8).reshape((-1, num_dims, 8)).flip(-1)

    # unpack bytes into bits (MSB first) and keep the num_bits low bits
    gray = (
        locs_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -num_bits:]
    )

    # Skilling transform: iterate forwards through bits and dimensions
    for bit in range(0, num_bits):
        for dim in range(0, num_dims):
            mask = gray[:, dim, bit]
            # where this bit is on, invert dimension 0's lower bits
            gray[:, 0, bit + 1:] = torch.logical_xor(
                gray[:, 0, bit + 1:], mask[:, None]
            )
            # where it is off, exchange the lower bits with dimension 0
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1:], gray[:, dim, bit + 1:]),
            )
            gray[:, dim, bit + 1:] = torch.logical_xor(gray[:, dim, bit + 1:], to_flip)
            gray[:, 0, bit + 1:] = torch.logical_xor(gray[:, 0, bit + 1:], to_flip)

    # interleave: (N, num_dims, num_bits) -> (N, num_bits * num_dims)
    gray = gray.swapaxes(1, 2).reshape((-1, num_bits * num_dims))

    # Gray code -> binary
    hh_bin = gray2binary(gray)

    # pad back out to 64 bits and pack into an int64
    extra_dims = 64 - num_bits * num_dims
    padded = torch.nn.functional.pad(hh_bin, (extra_dims, 0), "constant", 0)
    hh_uint8 = (
        (padded.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
        .sum(2)
        .to(torch.uint8)
    )
    hh_int64 = hh_uint8.contiguous().view(torch.int64).squeeze(-1)
    return hh_int64.reshape(orig_shape[:-1])


# ---------------------------------------------------------------------------
# Morton / z-order curve
# ---------------------------------------------------------------------------

def z_order_encode_(grid_coord, depth=16):
    """grid_coord (N, 3) ints < 2**depth -> (N,) int64 Morton codes (x at bit 0, y at 1, z at 2)."""
    x = grid_coord[:, 0].long()
    y = grid_coord[:, 1].long()
    z = grid_coord[:, 2].long()
    code = torch.zeros_like(x)
    for i in range(depth):
        code |= ((x >> i) & 1) << (3 * i + 0)
        code |= ((y >> i) & 1) << (3 * i + 1)
        code |= ((z >> i) & 1) << (3 * i + 2)
    return code


# ---------------------------------------------------------------------------
# Batch-major serialization codes
# ---------------------------------------------------------------------------

def encode(grid_coord, batch=None, depth=None, order="hilbert"):
    """Batch-major curve codes: ``batch << 3*depth | curve(grid_coord)`` -> (N,) int64,
    sorted = batch-contiguous space-filling order. `order` in ORDERS (*-trans swaps x/y);
    `depth` = bits/axis, from grid_coord.max() when None."""
    assert order in ORDERS, f"order must be one of {ORDERS}, got {order!r}"
    grid_coord = grid_coord.long()
    if depth is None:
        depth = max(int(grid_coord.max().item()).bit_length(), 1)
    assert depth * 3 <= 63, f"depth {depth} exceeds the int64 budget"

    if order in ("z-trans", "hilbert-trans"):
        grid_coord = grid_coord[:, [1, 0, 2]]

    if order in ("z", "z-trans"):
        code = z_order_encode_(grid_coord, depth=depth)
    else:
        code = hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)

    if batch is not None:
        batch = batch.long()
        assert depth * 3 + int(batch.max().item() + 1).bit_length() <= 63, \
            "batch id + curve code exceed the int64 budget; reduce depth or batch size"
        code = (batch << (depth * 3)) | code
    return code


def sort_and_inverse(code):
    """-> (order, inverse): x_sorted = x[order], x = x_sorted[inverse]."""
    order = torch.argsort(code)
    inverse = torch.zeros_like(order).scatter_(
        0, order, torch.arange(order.shape[0], device=order.device)
    )
    return order, inverse


def grid_coord_from_points(coord, grid_size):
    """coord (N, 3) float, grid_size metric voxel -> (N, 3) int64 grid coords shifted to start at 0."""
    grid_coord = torch.div(coord, grid_size, rounding_mode="floor").long()
    grid_coord = grid_coord - grid_coord.min(dim=0, keepdim=True)[0]
    return grid_coord
