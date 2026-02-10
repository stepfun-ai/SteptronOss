import time
from typing import TypedDict

import torch
from loguru import logger


def get_mem_brief(name="", unit="G"):
    """Simple GPU memory report."""
    giga_bytes = 1024**3
    m_bytes = 1024**2
    unit_bytes = m_bytes if unit == "M" else giga_bytes
    string = (
        f"{name} MemReport "
        f"Allocated: {torch.cuda.memory_allocated() / unit_bytes:.1f} "
        f"(max: {torch.cuda.max_memory_allocated() / unit_bytes:.1f}) {unit} | "
        f"Reserved: {torch.cuda.memory_reserved() / unit_bytes:.1f} "
        f"(max: {torch.cuda.max_memory_reserved() / unit_bytes:.1f}) {unit} | "
    )
    return string


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller context, use 0 for immediate caller.
        Default value: 0.
    Returns:
        str: module name of the caller
    """
    import inspect

    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe()
    # Skip current frame + caller depth.
    for _ in range(depth + 1):
        if frame is None:
            return "<unknown>"
        frame = frame.f_back
    if frame is None:
        return "<unknown>"
    return frame.f_globals.get("__name__", "<unknown>")


class MemoryRecord(TypedDict):
    mark: str
    time: float
    allocated: float
    reserved: float


class CudaMemoryTracker:
    def __init__(self):
        self.tracks: list[MemoryRecord] = []

    def mark(self, mark: str = None):
        if mark is None:
            mark = get_caller_name(depth=1)
        self.tracks.append(
            MemoryRecord(
                mark=mark,
                time=time.time(),
                allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),
            )
        )

    def report(self, topk: int = 3):
        """Report memory footprint collected, print these info:
        - Topk memory positions, marks and their time from start_time.
        - Topk memory delta, between which two marks mem gain most.

        and clear tracks.
        """
        if not self.tracks:
            logger.info("CudaMemoryTracker report: no records.")
            return

        topk = max(1, topk)

        def _fmt_time(ts: float) -> str:
            return time.strftime("%H:%M:%S", time.localtime(ts))

        def _fmt_bytes(num: float) -> str:
            return f"{num / (1024 ** 2):.1f} MB"

        # Topk memory positions by allocated bytes.
        top_positions = sorted(self.tracks, key=lambda x: x["allocated"], reverse=True)[:topk]

        lines = [
            "[CudaMemoryTracker Report]",
            f"Top{topk} memory positions (allocated):",
        ]
        for rec in top_positions:
            lines.append(
                "  "
                f"{rec['mark']} @ {_fmt_time(rec['time'])} "
                f"allocated={_fmt_bytes(rec['allocated'])}, "
                f"reserved={_fmt_bytes(rec['reserved'])}"
            )

        # Topk deltas between consecutive marks.
        if len(self.tracks) >= 2:
            deltas: list[tuple[float, int]] = []
            for idx in range(1, len(self.tracks)):
                prev = self.tracks[idx - 1]
                curr = self.tracks[idx]
                delta = curr["allocated"] - prev["allocated"]
                deltas.append((delta, idx))
            top_deltas = sorted(deltas, key=lambda x: x[0], reverse=True)[:topk]

            lines.append(f"Top{topk} memory deltas (allocated):")
            for delta, idx in top_deltas:
                prev = self.tracks[idx - 1]
                curr = self.tracks[idx]
                lines.append(
                    "  "
                    f"{prev['mark']} -> {curr['mark']} "
                    f"({ _fmt_time(prev['time'])} -> {_fmt_time(curr['time'])}) "
                    f"delta={_fmt_bytes(delta)}"
                )
        else:
            lines.append(f"Top{topk} memory deltas (allocated): insufficient records.")

        logger.info("\n".join(lines))
        self.tracks.clear()

    def report_over_world(self, topk: int = 3, topk_ranks: int = 1):
        """Like report, but not only in local, report {topk_ranks} ranks that have
        most and least memory peak. NOTE: this function use all_gather and therefore
        introduces a global barrier.
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            self.report(topk=topk)
            return

        topk = max(1, topk)
        topk_ranks = max(1, topk_ranks)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        def _fmt_time(ts: float) -> str:
            return time.strftime("%H:%M:%S", time.localtime(ts))

        def _fmt_bytes(num: float) -> str:
            return f"{num / (1024 ** 2):.1f} MB"

        lines = ["[CudaMemoryTracker Report]", f"Rank {rank}/{world_size}"]
        if not self.tracks:
            lines.append("No local records.")
            local_peak = MemoryRecord(
                mark="<empty>",
                time=0.0,
                allocated=0.0,
                reserved=0.0,
            )
        else:
            # Topk memory positions by allocated bytes.
            top_positions = sorted(self.tracks, key=lambda x: x["allocated"], reverse=True)[:topk]
            lines.append(f"Top{topk} memory positions (allocated):")
            for rec in top_positions:
                lines.append(
                    "  "
                    f"{rec['mark']} @ {_fmt_time(rec['time'])} "
                    f"allocated={_fmt_bytes(rec['allocated'])}, "
                    f"reserved={_fmt_bytes(rec['reserved'])}"
                )

            # Topk deltas between consecutive marks.
            if len(self.tracks) >= 2:
                deltas: list[tuple[float, int]] = []
                for idx in range(1, len(self.tracks)):
                    prev = self.tracks[idx - 1]
                    curr = self.tracks[idx]
                    delta = curr["allocated"] - prev["allocated"]
                    deltas.append((delta, idx))
                top_deltas = sorted(deltas, key=lambda x: x[0], reverse=True)[:topk]
                lines.append(f"Top{topk} memory deltas (allocated):")
                for delta, idx in top_deltas:
                    prev = self.tracks[idx - 1]
                    curr = self.tracks[idx]
                    lines.append(
                        "  "
                        f"{prev['mark']} -> {curr['mark']} "
                        f"({ _fmt_time(prev['time'])} -> {_fmt_time(curr['time'])}) "
                        f"delta={_fmt_bytes(delta)}"
                    )
            else:
                lines.append(f"Top{topk} memory deltas (allocated): insufficient records.")

            local_peak = max(self.tracks, key=lambda x: x["allocated"])

        from steptronoss.utils.dist_utils import all_gather_object

        payload = {
            "rank": rank,
            "mark": local_peak["mark"],
            "time": local_peak["time"],
            "allocated": float(local_peak["allocated"]),
            "reserved": float(local_peak["reserved"]),
        }
        gathered = all_gather_object(payload)

        if rank == 0:
            gathered_sorted = sorted(gathered, key=lambda x: x["allocated"])
            topk_ranks = min(topk_ranks, len(gathered_sorted))

            lines.append(f"Top{topk_ranks} ranks with least peak (allocated):")
            for rec in gathered_sorted[:topk_ranks]:
                lines.append(
                    "  "
                    f"rank={rec['rank']} "
                    f"{rec['mark']} @ {_fmt_time(rec['time'])} "
                    f"allocated={_fmt_bytes(rec['allocated'])}, "
                    f"reserved={_fmt_bytes(rec['reserved'])}"
                )

            lines.append(f"Top{topk_ranks} ranks with most peak (allocated):")
            for rec in gathered_sorted[-topk_ranks:][::-1]:
                lines.append(
                    "  "
                    f"rank={rec['rank']} "
                    f"{rec['mark']} @ {_fmt_time(rec['time'])} "
                    f"allocated={_fmt_bytes(rec['allocated'])}, "
                    f"reserved={_fmt_bytes(rec['reserved'])}"
                )

        logger.info("\n".join(lines), at=0)
        self.tracks.clear()


CMT = CudaMemoryTracker()
