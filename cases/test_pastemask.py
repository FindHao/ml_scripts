import time
import torch
import torch.nn.functional as F

import pickle
from torch import profiler

im_h = 427
im_w = 640


# pickle.dump(mask, open('/tmp/y1/mask', 'wb'))
# pickle.dump(box, open('/tmp/y1/box', 'wb'))

def paste_mask_in_image(mask, box, im_h, im_w):
    # type: (Tensor, Tensor, int, int) -> Tensor
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = F.interpolate(mask, size=(
        h, w), mode="bilinear", align_corners=False)
    mask = mask[0][0]

    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[(
        y_0 - box[1]): (y_1 - box[1]), (x_0 - box[0]): (x_1 - box[0])]
    return im_mask


def paste_mask_in_image_fast(mask, box, im_h, im_w, im_mask):
    # type: (Tensor, Tensor, int, int) -> Tensor
    TO_REMOVE = 1
    w = int(box[2] - box[0])
    w = w+TO_REMOVE if w > 0 else 1
    h = int(box[3] - box[1])
    h = h+TO_REMOVE if h > 0 else 1

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = F.interpolate(mask, size=(
        h, w), mode="bilinear", align_corners=False)
    mask = mask[0][0]

    # im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)
    x_0 = box[0] if box[0] > 0 else 0
    x_1 = box[2] + 1 if box[2] + 1 < im_w else im_w
    y_0 = box[1] if box[1] > 0 else 0
    y_1 = box[3] + 1 if box[3] + 1 < im_h else im_h

    im_mask[y_0:y_1, x_0:x_1] = mask[(
        y_0 - box[1]): (y_1 - box[1]), (x_0 - box[0]): (x_1 - box[0])]
    # return im_mask


def origin():
    return [paste_mask_in_image(m[0], b, im_h, im_w) for m, b in zip(masks, boxes)]


def opt():
    results = [torch.empty((im_h, im_w), dtype=mask[0].dtype,
                           device=mask[0].device) for mask in masks]
    torch._foreach_zero_(results)
    for i, pair in enumerate(zip(masks, boxes)):
        paste_mask_in_image_fast(pair[0][0], pair[1], im_h, im_w, results[i])

    return results


mask = pickle.load(open('/home/yhao/d/tmp/y1/mask', 'rb'))
box = pickle.load(open('/home/yhao/d/tmp/y1/box', 'rb'))

masks = pickle.load(open('/home/yhao/d/tmp/y1/masks', 'rb'))
boxes = pickle.load(open('/home/yhao/d/tmp/y1/boxes', 'rb'))


def check_consistency(res1, res2):
    import numpy as np
    if isinstance(res1, (tuple, list)):
        if not len(res1) == len(res2):
            return False

        return all(check_consistency(r1, r2) for r1, r2 in zip(res1, res2))

    assert isinstance(res1, np.ndarray)
    assert isinstance(res2, np.ndarray)
    return np.allclose(res1, res2)


def run():
    n_iter = 100
    for i in range(10):
        res = origin()
    t0 = time.time_ns()
    for i in range(n_iter):
        res = origin()
    # for i in range(n_iter):
    #     im_mask = paste_mask_in_image(mask, box, im_h, im_w)
    torch.cuda.synchronize()
    t1 = time.time_ns()
    duration = (t1 - t0) / 1_000_000
    print('{:<20} {:>20}'.format("Origin Total Wall Time:",
          "%.3f milliseconds" % (duration), sep=''))

    for i in range(10):
        res = opt()
    t0 = time.time_ns()
    # for i in range(n_iter):
    #     im_mask2 = paste_mask_in_image_fast(mask, box, im_h, im_w)
    for i in range(n_iter):
        res2 = opt()
    torch.cuda.synchronize()
    t1 = time.time_ns()
    duration2 = (t1 - t0) / 1_000_000
    print('{:<20} {:>20}'.format("Optimize Total Wall Time:",
          "%.3f milliseconds" % (duration2)), sep='')

    print("Speedup:", "%.3fX" % (duration / duration2))

    # if check_consistency(im_mask.cpu().detach().numpy(), im_mask2.cpu().detach().numpy()):
    #     print("Consistency Check Passed")
    ans1 = []
    for node in res:
        ans1.append(node.cpu().detach().numpy())
    ans2 = []
    for node in res2:
        ans2.append(node.cpu().detach().numpy())
    if check_consistency(ans1, ans2):
        print("Consistency Check Passed")
    else:
        print("Consistency Check Failed")


def profile():
    activity_groups = []
    activity_groups.append(profiler.ProfilerActivity.CUDA)
    activity_groups.append(profiler.ProfilerActivity.CPU)
    profile_detailed = True
    with profiler.profile(
        schedule=profiler.schedule(wait=0, warmup=3, active=1),
        activities=activity_groups,
        record_shapes=profile_detailed,
        profile_memory=profile_detailed,
        with_stack=profile_detailed,
        with_flops=profile_detailed,
        on_trace_ready=profiler.tensorboard_trace_handler('/tmp/logs2/')
    ) as prof:
        for i in range(4):
            opt()
            prof.step()


profile()
# run()

# paste_mask_in_image_fast(mask, box, im_h, im_w)
