from typing import Union, _alias, T
import torch
import numpy as np
import pickle
import time
List = _alias(list, T, inst=False)
import torch.profiler as profiler


def original_mask(polygons):
    def _make_array(t: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        # Use float64 for higher precision, because why not?
        # Always put polygons on CPU (self.to is a no-op) since they
        # are supposed to be small tensors.
        # May need to change this assumption if GPU placement becomes useful
        if isinstance(t, torch.Tensor):
            t = t.cpu().numpy()
        return np.asarray(t).astype("float64")

    def process_polygons(
        polygons_per_instance: List[Union[torch.Tensor, np.ndarray]]
    ) -> List[np.ndarray]:
        if not isinstance(polygons_per_instance, list):
            raise ValueError(
                "Cannot create polygons: Expect a list of polygons per instance. "
                "Got '{}' instead.".format(type(polygons_per_instance))
            )
        # transform each polygon to a numpy array
        polygons_per_instance = [_make_array(p) for p in polygons_per_instance]
        for polygon in polygons_per_instance:
            if len(polygon) % 2 != 0 or len(polygon) < 6:
                raise ValueError(f"Cannot create a polygon from {len(polygon)} coordinates.")
        return polygons_per_instance

    return [
        process_polygons(polygons_per_instance) for polygons_per_instance in polygons
    ]

def mask2(polygons):
    def _make_array(t: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        # Use float64 for higher precision, because why not?
        # Always put polygons on CPU (self.to is a no-op) since they
        # are supposed to be small tensors.
        # May need to change this assumption if GPU placement becomes useful
        if isinstance(t, torch.Tensor):
            t = t.cpu().numpy()
        return np.asarray(t).astype("float64")

    def process_polygons(
        polygons_per_instance: List[Union[torch.Tensor, np.ndarray]]
    ) -> List[np.ndarray]:
        if not isinstance(polygons_per_instance, list):
            raise ValueError(
                "Cannot create polygons: Expect a list of polygons per instance. "
                "Got '{}' instead.".format(type(polygons_per_instance))
            )
        # transform each polygon to a numpy array
        polygons_per_instance = [_make_array(p) for p in polygons_per_instance]
        for polygon in polygons_per_instance:
            if len(polygon) % 2 != 0 or len(polygon) < 6:
                raise ValueError(f"Cannot create a polygon from {len(polygon)} coordinates.")
        return polygons_per_instance

    return [
        process_polygons(polygons_per_instance) for polygons_per_instance in polygons
    ]




with open("/tmp/a.txt", 'rb') as fin:
    polygons = pickle.load(fin)


t0 = time.time_ns()
for i in range(100):
    original_mask(polygons)
t1 = time.time_ns()
print('{:<20} {:>20}'.format("Total Wall Time:", "%.3f milliseconds" % ((t1 - t0) / 1_000_000)), sep='')




# t0 = time.time_ns()
# activity_groups = []
# activity_groups.append(profiler.ProfilerActivity.CUDA)
# activity_groups.append(profiler.ProfilerActivity.CPU)
# profile_detailed=True
# with profiler.profile(
#         schedule=profiler.schedule(wait=0, warmup=0, active=1),
#         activities=activity_groups,
#         record_shapes=profile_detailed,
#         profile_memory=profile_detailed,
#         with_stack=profile_detailed,
#         with_flops=profile_detailed,
#         on_trace_ready=profiler.tensorboard_trace_handler("/tmp/logs/")
#     ) as prof:
#     original_mask(polygons)
# prof.export_chrome_trace("/tmp/a.pt.trace.json")
# t1 = time.time_ns()
# print('{:<20} {:>20}'.format("Total Wall Time:", "%.3f milliseconds" % ((t1 - t0) / 1_000_000)), sep='')

