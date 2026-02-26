from __future__ import annotations

import threading
from pathlib import Path
from queue import Empty, Queue
from typing import List, Optional

import torch

from tileflow.autotune import benchmark_io_and_compute, get_s_opt
from tileflow.config import RuntimeConfig


class ExpertTileStreamer:
    def __init__(self, tile_size_mb: int, prefetch_depth: int, device: torch.device):
        self.tile_size_mb = tile_size_mb
        self.tile_size_bytes = tile_size_mb * 1024 * 1024
        self.prefetch_depth = prefetch_depth
        self.device = device
        self.prefetch_queue: Queue[tuple[int, torch.Tensor]] = Queue(maxsize=prefetch_depth)
        self.stop_event = threading.Event()
        self.prefetch_thread: Optional[threading.Thread] = None

    def _prefetch_worker(self, tile_paths: List[Path]) -> None:
        tile_idx = 0
        while not self.stop_event.is_set() and tile_idx < len(tile_paths):
            num_elements = self.tile_size_bytes // 2
            cpu_tensor = torch.empty(num_elements, dtype=torch.bfloat16, pin_memory=True)
            gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)
            self.prefetch_queue.put((tile_idx, gpu_tensor), block=True)
            tile_idx += 1

    def start_prefetching(self, tile_paths: List[Path]) -> None:
        self.stop_event.clear()
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(tile_paths,),
            daemon=True,
        )
        self.prefetch_thread.start()

    def get_next_tile(self, timeout: float = 10.0) -> Optional[tuple[int, torch.Tensor]]:
        try:
            return self.prefetch_queue.get(timeout=timeout)
        except Empty:
            return None

    def stop(self) -> None:
        self.stop_event.set()
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=2.0)
        while not self.prefetch_queue.empty():
            try:
                _, tensor = self.prefetch_queue.get_nowait()
                del tensor
            except Empty:
                break


class StreamingMoEInference:
    def __init__(self, tile_size_mb: int, prefetch_depth: int, hidden_dim: int = 7168, device_id: int = 0):
        self.device = torch.device(f"cuda:{device_id}")
        self.hidden_dim = hidden_dim
        self.streamer = ExpertTileStreamer(tile_size_mb, prefetch_depth, self.device)

    def process_tile(self, tile_data: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
        # Demonstration-only compute placeholder.
        result = torch.matmul(activation, tile_data[: self.hidden_dim, :])
        result = torch.relu(result)
        result = torch.matmul(result, tile_data[: self.hidden_dim, :].t())
        return result

    def run_inference(self, tile_paths: List[Path], num_tokens: int = 10) -> None:
        self.streamer.start_prefetching(tile_paths)
        torch.cuda.synchronize()
        try:
            for _ in range(num_tokens):
                activation = torch.randn(1, self.hidden_dim, dtype=torch.bfloat16, device=self.device)
                outputs = []
                for _ in range(len(tile_paths)):
                    tile_result = self.streamer.get_next_tile()
                    if tile_result is None:
                        continue
                    _, tile_data = tile_result
                    outputs.append(self.process_tile(tile_data, activation))
                    del tile_data
                if outputs:
                    final_output = torch.cat(outputs, dim=-1)
                    del final_output
                torch.cuda.synchronize()
        finally:
            self.streamer.stop()


def demo_workflow() -> None:
    cfg = RuntimeConfig()
    bench = benchmark_io_and_compute(cfg)
    tune = get_s_opt(bench.io_bandwidth_gbs, bench.compute_times_s, cfg)
    tile_paths = [Path(f"/tmp/expert_tile_{i}.pt") for i in range(8)]
    inference = StreamingMoEInference(tile_size_mb=tune.s_opt_mb, prefetch_depth=tune.prefetch_depth)
    inference.run_inference(tile_paths, num_tokens=5)

