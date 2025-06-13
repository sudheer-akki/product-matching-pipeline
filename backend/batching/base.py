import asyncio
import inspect
from typing import Any, Callable, List, Tuple

class InferenceBatcher:
    """
    Collects individual inference requests and batches them
    for efficient processing via Triton or local inference.

    Supports:
    - Max batch size based dispatch
    - Max wait time based dispatch
    - Async result delivery
    """

    def __init__(
        self,
        max_batch_size: int,
        max_wait_time: float,
        infer_fn: Callable[[List[Any]], List[Any]],
        mode: str = "auto"
    ):
        """
        Args:
            model_name (str): Name of the model (for logging/trace).
            max_batch_size (int): Maximum number of items per batch.
            max_wait_time (float): Max wait (in seconds) before forcing a batch.
            infer_fn (Callable): Function that takes List[inputs] and returns List[outputs].
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.infer_fn = infer_fn
        self.mode = mode

        self._queue: asyncio.Queue[Tuple[Any, asyncio.Future]] = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._batch_worker())

    async def submit(self, input_item: dict) -> Any:
        """
        Submit an inference request. Returns the output via future.
        """
        if inspect.iscoroutine(input_item):
            raise ValueError("input_item is a coroutine")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put((input_item, future))
        return await future


    async def _batch_worker(self):
        while True:
            try:
                batch_inputs, futures = await self._collect_batch()
                outputs = await self._run_inference(batch_inputs)
                self._resolve_futures(futures, outputs)
            except Exception as e:
                self._fail_futures(futures, e)


    async def _collect_batch(self) -> Tuple[List[Any], List[asyncio.Future]]:
            """
            Collects inputs until max_batch_size is reached or timeout occurs.
            """
            batch_inputs = []
            futures = []

            item, fut = await self._queue.get()
            batch_inputs.append(item)
            futures.append(fut)

            start_time = asyncio.get_event_loop().time()
            while len(batch_inputs) < self.max_batch_size:
                elapsed = asyncio.get_event_loop().time() - start_time
                timeout = self.max_wait_time - elapsed
                if timeout <= 0:
                    break
                try:
                    item, fut = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                    batch_inputs.append(item)
                    futures.append(fut)
                except asyncio.TimeoutError:
                    break

            return batch_inputs, futures
    
    async def _run_inference(self, batch_inputs: List[Any]) -> List[Any]:
        if self.mode == "async":
            # Always await, fail if not async
            result = self.infer_fn(batch_inputs)
            if not inspect.isawaitable(result):
                raise RuntimeError("infer_fn must be async when mode='async'")
            return await result
        elif self.mode == "sync":
            # Always run in executor, even if it’s awaitable (force sync)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.infer_fn, batch_inputs)
        else:  # auto mode (default, your current logic)
            result = self.infer_fn(batch_inputs)
            if inspect.isawaitable(result):
                return await result
            else:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, self.infer_fn, batch_inputs)

    def _resolve_futures(self, futures: List[asyncio.Future], outputs: List[Any]):
        """
        Sends successful inference outputs back to each future.
        """
        for fut, output in zip(futures, outputs):
            if not fut.done():
                fut.set_result(output)

    def _fail_futures(self, futures: List[asyncio.Future], error: Exception):
        """
        Handles exception case by propagating error to pending futures.
        """
        for fut in futures:
            if not fut.done():
                fut.set_exception(error)