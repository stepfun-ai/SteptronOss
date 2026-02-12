from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait

from loguru import logger


class TimedRunner:
    def __init__(self):
        self.executor = ProcessPoolExecutor(max_workers=1)

    def __del__(self):
        self.executor.shutdown()

    def reset(self, *exc_info):
        try:
            # Use kill instead of terminate for stubborn processes
            for _pid, proc in self.executor._processes.items():
                try:
                    proc.kill()  # More forceful than terminate
                except:
                    pass  # Ignore errors from killing processes

            # Add timeout to shutdown
            self.executor.shutdown(wait=False)
        except Exception as e:
            # Log exception but continue
            from loguru import logger

            logger.exception(f"Error during executor reset: {e}")
        finally:
            # Always create a new executor
            self.executor = ProcessPoolExecutor(max_workers=1)

    def __call__(self, func, *args, timeout=10, **kwargs):
        from loguru import logger

        try:
            future = self.executor.submit(func, *args, **kwargs)
            done, not_done = wait({future}, timeout=timeout, return_when=FIRST_EXCEPTION)
        except:
            logger.warning("Executor broken, reset executor...")
            self.reset()
            future = self.executor.submit(func, *args, **kwargs)
            done, not_done = wait({future}, timeout=timeout, return_when=FIRST_EXCEPTION)

        if not_done:
            logger.warning("Executor timeout, reset executor...")
            self.reset()
            raise TimeoutError(f"Runner timeout after {timeout} seconds")

        res = future.result(timeout=0)
        return res


_RUNNER: TimedRunner = None


def get_runner() -> TimedRunner:
    global _RUNNER
    if _RUNNER is None:
        _RUNNER = TimedRunner()
    return _RUNNER
