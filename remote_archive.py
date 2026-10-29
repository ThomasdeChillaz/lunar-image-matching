"""Read selected ZIP members over verified HTTP ranges, including ZIP64 archives.

No extraction paths are trusted. zipfile checks each downloaded member's CRC;
callers should also preserve the archive record and hash the selected bytes.
"""
import io
from collections import OrderedDict
import hashlib
from pathlib import Path
import time
import urllib.request
import zipfile


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url, size, block_size=1024*1024, max_read=64*1024*1024, cache_dir=None):
        self.url, self.size = url, int(size)
        self.block_size, self.max_read = block_size, max_read
        self.position, self.cache, self.downloaded_bytes = 0, OrderedDict(), 0
        self.disk_cache = None
        if cache_dir is not None:
            self.disk_cache = Path(cache_dir)/hashlib.sha256((url+':'+str(size)).encode()).hexdigest()
            self.disk_cache.mkdir(parents=True, exist_ok=True)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == 0 else (self.position if whence == 1 else self.size)+offset
        if position < 0:
            raise ValueError('Negative seek')
        self.position = position
        return position

    def read(self, size=-1):
        size = min(self.size-self.position, size if size >= 0 else self.size-self.position)
        if size <= 0:
            return b''
        if size > self.max_read:
            raise ValueError('Requested range exceeds per-read safety limit')
        end, chunks = self.position+size, []
        while self.position < end:
            block = self.position//self.block_size
            start = block*self.block_size
            if block not in self.cache:
                stop = min(start+self.block_size, self.size)-1
                disk = self.disk_cache/f'{start}-{stop}.bin' if self.disk_cache else None
                checksum = disk.with_suffix('.sha256') if disk else None
                data = None
                if disk and disk.exists() and checksum.exists():
                    content = disk.read_bytes()
                    if len(content) == stop-start+1 and hashlib.sha256(content).hexdigest() == checksum.read_text():
                        data = content
                if data is None:
                    for attempt in range(3):
                        req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{stop}', 'Accept-Encoding': 'identity'})
                        try:
                            with urllib.request.urlopen(req, timeout=45) as response:
                                expected = f'bytes {start}-{stop}/{self.size}'
                                if response.status != 206 or response.headers.get('Content-Range') != expected:
                                    raise ValueError('Server did not honor exact byte range; refusing full archive download')
                                block_chunks, remaining = [], stop-start+1
                                while remaining:
                                    chunk = response.read(min(remaining, 1024*1024))
                                    if not chunk:
                                        raise OSError('Truncated byte range')
                                    block_chunks.append(chunk)
                                    remaining -= len(chunk)
                                data = b''.join(block_chunks)
                            break
                        except (OSError, TimeoutError):
                            if attempt == 2:
                                raise
                            print(f'Retrying archive bytes {start}-{stop} after interrupted transfer', flush=True)
                            time.sleep(2**attempt)
                    if disk:
                        temporary = disk.with_suffix('.part')
                        temporary.write_bytes(data)
                        temporary.replace(disk)
                        checksum.write_text(hashlib.sha256(data).hexdigest())
                    self.downloaded_bytes += len(data)
                self.cache[block] = data
                if len(self.cache) > 8:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(block)
            take = min(end-self.position, len(self.cache[block])-(self.position-start))
            chunks.append(self.cache[block][self.position-start:self.position-start+take])
            self.position += take
        return b''.join(chunks)


def open_remote_zip(url, size):
    reader = HTTPRangeReader(url, size)
    return zipfile.ZipFile(reader), reader
