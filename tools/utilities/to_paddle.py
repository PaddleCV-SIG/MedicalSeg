
import paddle


def maybe_to_paddle(d):
    if isinstance(d, list):
        d = [maybe_to_paddle(i) if not isinstance(i, paddle.Tensor) else i for i in d]
    elif not isinstance(d, paddle.Tensor):
        d = paddle.to_tensor(d,"float32")
    return d


def to_cuda(data, non_blocking=True, gpu_id=0):

    return data
