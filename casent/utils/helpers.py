import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from accelerate import Accelerator
from dataclasses import dataclass
import asyncio
import aiohttp
from tqdm import tqdm
import re
from typing import Tuple, List, Callable, Awaitable, Optional, Any
import nltk
import collections
import argparse
import os


def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def check_path(path):
    check_dir(os.path.dirname(path))


def bool_flag(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def pack_list(l: list) -> Tuple[list, list]:
    """Given a list, returns a list of its unique elements and the counts"""
    counter = collections.Counter(l)
    counter = sorted(counter.items(), key=lambda t: -t[1])
    l_unique, counts = list(zip(*counter))
    return l_unique, counts


@dataclass
class ParallelHTTPClient:
    """
    An efficient http client based on aiohttp that handles batch requests
    with rate/parallelism limiting, retrying and response post-processing.
    """
    max_retries: Optional[int] = None
    max_tcp_conns: int = 100
    max_request_per_second: int = 10
    async_process_fn: Callable[[aiohttp.ClientResponse], Awaitable] = aiohttp.ClientResponse.text
    verbose: bool = True
    progress_bar: bool = True

    def __post_init__(self):
        self._queue = []
        self._tasks = []
        self._results = []
        self._fail_counts = None
        self._session = None
        self._pbar = None

    def get(self, url: str):
        _progress_bar, self.progress_bar = self.progress_bar, False  # temporarily disable progress bar
        res = self.batch_get([url])[0]
        self.progress_bar = _progress_bar
        return res

    def batch_get(self, url_list: List[str]):
        return asyncio.run(self._async_batch_get(url_list))

    async def _async_batch_get(self, url_list: List[str]) -> List[Any]:
        connector = aiohttp.TCPConnector(limit=self.max_tcp_conns)
        self._queue = [(i, url) for i, url in enumerate(url_list)]
        self._tasks = []
        self._results = [None] * len(url_list)
        self._fail_counts = collections.defaultdict(int)
        with tqdm(total=len(url_list), disable=not self.progress_bar) as self._pbar:
            async with aiohttp.ClientSession(connector=connector) as self._session:
                while True:
                    while len(self._queue) > 0:
                        url_id, url = self._queue.pop(0)
                        self._tasks.append(asyncio.create_task(self._fetch(url_id, url)))
                        await asyncio.sleep(1 / self.max_request_per_second)
                    await asyncio.gather(*self._tasks)
                    self._tasks = []
                    if len(self._queue) == 0:
                        break
        await connector.close()
        return self._results

    async def _fetch(self, url_id, url):
        try:
            async with self._session.get(url=url) as response:
                assert response.status == 200, f'Response http status {response.status}'
                self._pbar.update(1)
                self._results[url_id] = await self.async_process_fn(response)
        except (aiohttp.ClientError, AssertionError) as e:
            if self.verbose:
                print(f'URL {url} encountered error: {e}')
            if self.max_retries is not None and self._fail_counts[url_id] == self.max_retries:
                print(f'URL {url} failed')
            else:
                self._fail_counts[url_id] += 1
                self._queue.append((url_id, url))


def test():
    urls = ['https://www.wikidata.org/wiki/Q55183727',
            'https://www.wikidata.org/wiki/Q5']
    client = ParallelHTTPClient(max_retries=1)
    res = client.batch_get(urls)
    res = client.get(urls[0])
    client.get('https://www.asdfasfdads.com')


if __name__ == '__main__':
    test()
