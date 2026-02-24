import atexit
import os
import time
from typing import TypedDict

import torch
from loguru import logger

from steptronoss.utils.general import convert_num


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
    cpu_rss: float


def _get_cpu_rss_bytes() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return float(parts[1]) * 1024
    except Exception:
        pass
    try:
        import resource

        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return 0.0


class CudaMemoryTracker:
    def __init__(self):
        self.tracks: list[MemoryRecord] = []

    def mark(self, mark: str | None = None):
        if os.getenv("MEM_DIAGNOSE") is None:
            return
        if mark is None:
            mark = get_caller_name(depth=1)
        allocated = 0.0
        reserved = 0.0
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
        self.tracks.append(
            MemoryRecord(
                mark=mark,
                time=time.time(),
                allocated=allocated,
                reserved=reserved,
                cpu_rss=_get_cpu_rss_bytes(),
            )
        )

    def report(self, topk: int = 3):
        """Report memory footprint collected, print these info:
        - Topk memory positions, marks and their time from start_time.
        - Topk memory delta, between which two marks mem gain most.

        and clear tracks.
        """
        if os.getenv("MEM_DIAGNOSE") is None:
            return
        if not self.tracks:
            logger.info("CudaMemoryTracker report: no records.")
            return

        topk = max(1, topk)

        def _fmt_time(ts: float) -> str:
            return time.strftime("%H:%M:%S", time.localtime(ts))

        def _fmt_bytes(num: float) -> str:
            return f"{convert_num(num, G=True)}B"

        # Topk CUDA memory positions by allocated bytes.
        top_positions = sorted(self.tracks, key=lambda x: x["allocated"], reverse=True)[:topk]

        lines = [
            "[CudaMemoryTracker Report]",
            f"Top{topk} CUDA memory positions (allocated):",
        ]
        for rec in top_positions:
            lines.append(
                "  "
                f"{rec['mark']} @ {_fmt_time(rec['time'])} "
                f"allocated={_fmt_bytes(rec['allocated'])}, "
                f"reserved={_fmt_bytes(rec['reserved'])}"
            )

        # Topk CUDA deltas between consecutive marks.
        if len(self.tracks) >= 2:
            deltas: list[tuple[float, int]] = []
            for idx in range(1, len(self.tracks)):
                prev = self.tracks[idx - 1]
                curr = self.tracks[idx]
                delta = curr["allocated"] - prev["allocated"]
                deltas.append((delta, idx))
            top_deltas = sorted(deltas, key=lambda x: x[0], reverse=True)[:topk]

            lines.append(f"Top{topk} CUDA memory deltas (allocated):")
            for delta, idx in top_deltas:
                prev = self.tracks[idx - 1]
                curr = self.tracks[idx]
                lines.append(
                    "  "
                    f"{prev['mark']} -> {curr['mark']} "
                    f"({_fmt_time(prev['time'])} -> {_fmt_time(curr['time'])}) "
                    f"delta={_fmt_bytes(delta)}"
                )
        else:
            lines.append(f"Top{topk} CUDA memory deltas (allocated): insufficient records.")

        # Topk CPU memory positions by rss bytes.
        top_cpu_positions = sorted(self.tracks, key=lambda x: x["cpu_rss"], reverse=True)[:topk]
        lines.append(f"Top{topk} CPU memory positions (rss):")
        for rec in top_cpu_positions:
            lines.append(f"  {rec['mark']} @ {_fmt_time(rec['time'])} rss={_fmt_bytes(rec['cpu_rss'])}")

        # Topk CPU deltas between consecutive marks.
        if len(self.tracks) >= 2:
            cpu_deltas: list[tuple[float, int]] = []
            for idx in range(1, len(self.tracks)):
                prev = self.tracks[idx - 1]
                curr = self.tracks[idx]
                delta = curr["cpu_rss"] - prev["cpu_rss"]
                cpu_deltas.append((delta, idx))
            top_cpu_deltas = sorted(cpu_deltas, key=lambda x: x[0], reverse=True)[:topk]
            lines.append(f"Top{topk} CPU memory deltas (rss):")
            for delta, idx in top_cpu_deltas:
                prev = self.tracks[idx - 1]
                curr = self.tracks[idx]
                lines.append(
                    "  "
                    f"{prev['mark']} -> {curr['mark']} "
                    f"({_fmt_time(prev['time'])} -> {_fmt_time(curr['time'])}) "
                    f"delta={_fmt_bytes(delta)}"
                )
        else:
            lines.append(f"Top{topk} CPU memory deltas (rss): insufficient records.")

        logger.info("\n".join(lines))
        self.tracks.clear()

    def report_over_world(self, topk: int = 3, topk_ranks: int = 1):
        """Like report, but not only in local, report {topk_ranks} ranks that have
        most and least memory peak. NOTE: this function use all_gather and therefore
        introduces a global barrier.
        """
        if os.getenv("MEM_DIAGNOSE") is None:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            self.report(topk=topk)
            return

        topk = max(1, topk)
        topk_ranks = max(1, topk_ranks)
        rank = torch.distributed.get_rank()

        def _fmt_time(ts: float) -> str:
            return time.strftime("%H:%M:%S", time.localtime(ts))

        def _fmt_bytes(num: float) -> str:
            return f"{convert_num(num, G=True)}B"

        lines = ["[CudaMemoryTracker Report]", "=" * 90]
        if not self.tracks:
            lines.append("No local records.")
            local_peak = MemoryRecord(
                mark="<empty>",
                time=0.0,
                allocated=0.0,
                reserved=0.0,
                cpu_rss=0.0,
            )
        else:
            # Topk CUDA memory positions by allocated bytes.
            top_positions = sorted(self.tracks, key=lambda x: x["allocated"], reverse=True)[:topk]
            lines.append(f"Top{topk} CUDA memory positions (allocated):")
            for rec in top_positions:
                lines.append(
                    "  "
                    f"{rec['mark']} @ {_fmt_time(rec['time'])} "
                    f"allocated={_fmt_bytes(rec['allocated'])}, "
                    f"reserved={_fmt_bytes(rec['reserved'])}"
                )

            # Topk CUDA deltas between consecutive marks.
            if len(self.tracks) >= 2:
                deltas: list[tuple[float, int]] = []
                for idx in range(1, len(self.tracks)):
                    prev = self.tracks[idx - 1]
                    curr = self.tracks[idx]
                    delta = curr["allocated"] - prev["allocated"]
                    deltas.append((delta, idx))
                top_deltas = sorted(deltas, key=lambda x: x[0], reverse=True)[:topk]
                lines.append(f"Top{topk} CUDA memory deltas (allocated):")
                for delta, idx in top_deltas:
                    prev = self.tracks[idx - 1]
                    curr = self.tracks[idx]
                    lines.append(
                        "  "
                        f"{prev['mark']} -> {curr['mark']} "
                        f"({_fmt_time(prev['time'])} -> {_fmt_time(curr['time'])}) "
                        f"delta={_fmt_bytes(delta)}"
                    )
            else:
                lines.append(f"Top{topk} CUDA memory deltas (allocated): insufficient records.")

            top_cpu_positions = sorted(self.tracks, key=lambda x: x["cpu_rss"], reverse=True)[:topk]
            lines.append(f"Top{topk} CPU memory positions (rss):")
            for rec in top_cpu_positions:
                lines.append(f"  {rec['mark']} @ {_fmt_time(rec['time'])} rss={_fmt_bytes(rec['cpu_rss'])}")

            if len(self.tracks) >= 2:
                cpu_deltas: list[tuple[float, int]] = []
                for idx in range(1, len(self.tracks)):
                    prev = self.tracks[idx - 1]
                    curr = self.tracks[idx]
                    delta = curr["cpu_rss"] - prev["cpu_rss"]
                    cpu_deltas.append((delta, idx))
                top_cpu_deltas = sorted(cpu_deltas, key=lambda x: x[0], reverse=True)[:topk]
                lines.append(f"Top{topk} CPU memory deltas (rss):")
                for delta, idx in top_cpu_deltas:
                    prev = self.tracks[idx - 1]
                    curr = self.tracks[idx]
                    lines.append(
                        "  "
                        f"{prev['mark']} -> {curr['mark']} "
                        f"({_fmt_time(prev['time'])} -> {_fmt_time(curr['time'])}) "
                        f"delta={_fmt_bytes(delta)}"
                    )
            else:
                lines.append(f"Top{topk} CPU memory deltas (rss): insufficient records.")

            local_peak = max(self.tracks, key=lambda x: x["allocated"])

        from steptronoss.utils.dist_utils import all_gather_object

        payload = {
            "rank": rank,
            "mark": local_peak["mark"],
            "time": local_peak["time"],
            "allocated": float(local_peak["allocated"]),
            "reserved": float(local_peak["reserved"]),
            "cpu_rss": float(local_peak["cpu_rss"]),
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
            gathered_sorted_cpu = sorted(gathered, key=lambda x: x["cpu_rss"])
            lines.append(f"Top{topk_ranks} ranks with least CPU peak (rss):")
            for rec in gathered_sorted_cpu[:topk_ranks]:
                lines.append(
                    f"  rank={rec['rank']} {rec['mark']} @ {_fmt_time(rec['time'])} rss={_fmt_bytes(rec['cpu_rss'])}"
                )
            lines.append(f"Top{topk_ranks} ranks with most CPU peak (rss):")
            for rec in gathered_sorted_cpu[-topk_ranks:][::-1]:
                lines.append(
                    f"  rank={rec['rank']} {rec['mark']} @ {_fmt_time(rec['time'])} rss={_fmt_bytes(rec['cpu_rss'])}"
                )
        lines.append("=" * 90)
        logger.info("\n".join(lines), at=0)
        self.tracks.clear()


CMT = CudaMemoryTracker()


def _report_cmt_at_exit():
    try:
        CMT.mark("on_exit")
        CMT.report()
    except Exception:
        # Best-effort on shutdown; avoid raising during interpreter exit.
        pass


atexit.register(_report_cmt_at_exit)
