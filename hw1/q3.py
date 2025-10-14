###Q3: allreduce###
###please implement ring_allreduce method, using  pytorch's dist method is not allowed###

from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.distributed as dist

def reduce_scatter(chunks, tmp, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    chunk_sz = chunks[0].numel()
    buf = torch.empty(chunk_sz, dtype=chunks[0].dtype, device=chunks[0].device)

    for s in range(world - 1):
        send_idx = (rank - s) % world
        recv_idx = (rank - s - 1) % world

        send_req = dist.isend(chunks[send_idx], dst=right)
        recv_req = dist.irecv(buf, src=left)
        recv_req.wait()
        chunks[recv_idx].add_(buf)
        send_req.wait()

    current = (rank - (world - 1)) % world
    return current
        
def all_gather(chunks, tmp, current, world, rank, left, right):
    #                                                                   #
    #                                                                   #
    # your code here: follow slides instruction: do counter-clockwise iteration
    #                                                                   #
    #                                                                   #
    chunk_sz = chunks[0].numel()
    for s in range(world - 1):
        next_idx = (current - 1) % world
        send_req = dist.isend(chunks[current], dst=right)
        recv_req = dist.irecv(chunks[next_idx], src=left)  # write directly
        recv_req.wait()
        send_req.wait()
        current = next_idx

def ring_allreduce_(tensor: torch.Tensor, world_size = None, rankid = None):
    """In-place ring all-reduce (SUM, optional average) using isend/irecv."""
    world = world_size
    if world == 1: return tensor
    rank = rankid
    left, right = (rank - 1) % world, (rank + 1) % world

    ##following steps try to fill blank to the tensor so that final tensor can be divided to 3 chunks evenly
    flat = tensor.contiguous().view(-1)
    n = flat.numel()
    chunk = (n + world - 1) // world
    #                                                                   #
    #                                                                   #
    # your code here: we cannot divide flat into 3 pieces evenly as the
    # flat lengh may not be able to divided exactly by 3....
    #
    #                                                                   #
    #                                                                   #
    #So, fill zeros at the end of flat to generate padded_flat
    padded_n = chunk * world
    padded_flat = torch.zeros(
        padded_n, dtype=flat.dtype, device=flat.device
    ) # modify this line and fill correct value into padded_flat
    chunks = [padded_flat[i*chunk:(i+1)*chunk] for i in range(world)]

    #                                                                   #
    #                                                                   #
    # your code here: call reduce_scatter and all_gather
    #
    #                                                                   #
    #                                                                   #
    #we provide the reduce_scatter and all_gather func prototype for you
    # You may adjust the function signature (input structure) of `reduce_scatter` and `all_gather` if needed.
    current = reduce_scatter(chunks, None, world, rank, left, right)
    all_gather(chunks, None, current, world, rank, left, right)
    flat.copy_(padded_flat[:n])
    
    # stitch & unpad  
    flat /= world
    tensor.view(-1).copy_(flat[:n])
    return