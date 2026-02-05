import contextlib
import pickle
import time
from hashlib import md5
from threading import Thread

from loguru import logger
from megfile import smart_exists, smart_open, smart_path_join


class Block:
    def __init__(self, root="", target_file=None, checksum="EMPTY", next=None, verify=True):

        self.root = root
        self.target_file = target_file
        self.checksum = checksum
        self.loaded = False

        if next is None or next == []:
            self.next = [None, "EMPTY"]
        else:
            self.next = next
        self.verify = verify

        self.data = []

        if root and not smart_exists(self.target_path):
            self.save()

    @property
    def target_path(self):
        return smart_path_join(self.root, self.target_file)

    def md5(self):
        return md5(pickle.dumps(self.data)).hexdigest()

    def load(self):
        with smart_open(self.target_path, "rb") as f:
            self.data, self.next = pickle.load(f)
        if self.verify and self.data:
            new_checksum = self.md5()
            if new_checksum != self.checksum:
                logger.warning(f"Checksum mismatch on {self.target_path}. Double check if this is expected.")
                self.checksum = new_checksum
        self.loaded = True

    def unload(self):
        logger.info(f"Block {self.target_file}'s data released.")
        del self.data
        self.loaded = False

    def __len__(self):
        return len(self.data)

    def state_dict(self):
        return dict(
            target_file=self.target_file,
            checksum=self.checksum,
            next=self.next,
        )

    def save(self, file=None):
        if file:
            self.target_file = file
        with smart_open(self.target_path, "wb") as f:
            pickle.dump([self.data, self.next], f)


class Chain:
    def __init__(self, root="", verify=True, prefetch=0, verbose=True, reusable=False):
        """Portable file-chain for data transfer and feeding.

        Args:
            root (str, optional): dir to the chain files, create if not exist. Defaults to ''.
            verify (bool, optional): verify checksum. Defaults to True.
            prefetch (int, optional): number of prefetch blocks. Defaults to 0.
            verbose (bool, optional): Defaults to False.

        Use:
            chain = Chain('/my/path/aaa/')
            with chain.writer(block_size=1000) as write:
                for i in data:
                    write(i)

            !cp /my/path/aaa/ /my/path/bbb/

            chain = Chain('/my/path/bbb/')
            while i:= chain.get():
                print(i)
        """
        self._root = root

        root_block = Block(self.root, "root.pkl")
        self.blocks: list[Block] = [root_block]
        self.block_id = 0
        self.in_block_id = 0

        self.ended = False

        self.verify = verify

        self.verbose = verbose
        self.reusable = reusable

        self.prefetch = prefetch
        self.pre_count = 0
        self.loader = None

    def run_loader(self):
        if not self.loader:
            self.loader = Thread(target=self._loader, daemon=True)
            self.loader.start()

    def _loader(self):
        while self.prefetch:
            self.pre_count = 0
            block_id = self.block_id
            while self.pre_count <= self.prefetch:
                block = self.blocks[block_id + self.pre_count]
                if not block.loaded:
                    block.load()
                    if self.verbose:
                        logger.info(f"Prefetch -> {block.target_file}")
                if block_id + self.pre_count + 1 >= len(self.blocks):
                    if block.next[0] is None:
                        break
                    self.blocks.append(Block(self.root, *block.next, verify=self.verify))
                self.pre_count += 1
            time.sleep(0.1)

    def load_block(self, block: Block):
        if not block.loaded:
            if self.verbose:
                logger.info(f"FetchBlock -> {block.target_file}")
            if self.loader and self.loader.is_alive():
                while not block.loaded:
                    time.sleep(0.1)
            else:
                block.load()

    def fastload(self):
        def get_blocks(dir):
            import os
            import re

            files = [(f, int(re.findall(r"block_(\d+)_", f)[0])) for f in os.listdir(dir) if f.startswith("block_")]
            files.sort(key=lambda x: x[1])
            return files

        self.verify = False
        assert len(self.blocks) == 1
        ct = 1
        for f, idx in get_blocks(self.root):
            assert idx == ct, f"block {ct} mismatch!"
            self.blocks.append(Block(self.root, f, verify=False))
            ct += 1
        print("Fastload Success!")

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        self._root = value
        for block in self.blocks:
            block.root = value

    def get_nostep(self):
        block = self.blocks[self.block_id]
        self.load_block(block)

        # handle emply block
        if self.in_block_id >= len(block.data):
            logger.warning(f"Empty: {block.target_file}")
            self.next()
            block = self.blocks[self.block_id]
            self.load_block(block)

        return block.data[self.in_block_id]

    def get(self):
        if self.ended:
            if not self.next():
                logger.warning("Get Nothing!")
                return None
        data = self.get_nostep()
        self.next()
        return data

    def next(self):
        if self.in_block_id + 1 >= len(self.blocks[self.block_id]):
            # next block
            if self.ended:
                self.blocks[self.block_id].load()
            next_block = self.blocks[self.block_id].next
            if next_block == [] or next_block[0] is None:
                print(f"{len(self.blocks)=}. next_block is empty, restart from root!!!")
                logger.warning(f"{len(self.blocks)=}. next_block is empty, restart from root!!!")
                maybe_next_file, maybe_next_checksum = "root.pkl", ""
            else:
                maybe_next_file, maybe_next_checksum = next_block

            # maybe_next_file, maybe_next_checksum = self.blocks[self.block_id].next

            # if not reach end of self.blocks, use next block in list
            # else, try locate next block using block.next
            if self.block_id + 1 >= len(self.blocks):
                if maybe_next_file is not None:
                    next_block = Block(
                        self.root,
                        maybe_next_file,
                        maybe_next_checksum,
                        verify=self.verify,
                    )
                    self.blocks.append(next_block)
                else:
                    self.ended = True
                    logger.warning("Reach end of the chain!")
                    return False

            if not self.reusable:
                self.blocks[self.block_id].unload()
                self.pre_count -= 1
            self.block_id += 1
            self.in_block_id = 0
        else:
            self.in_block_id += 1
        self.ended = False
        return True

    def extend(self):
        full = False
        while not full:
            block = self.blocks[-1]
            try:
                block.load()
                print(block.target_file, block.checksum)
                full = self.blocks[-1].next[0] is None
            except:
                full = True
            if full:
                break
            self.blocks.append(Block(self.root, *block.next, verify=self.verify))
            block.unload()

    def validate(self):
        block = self.blocks[0]
        if not block.loaded:
            block.load()
        print(f"Root: {block.target_file} [{len(block.data)}]")
        block.unload()
        next_b = block.next

        for bid in range(1, len(self.blocks)):
            block = self.blocks[bid]
            if not block.loaded:
                block.load()
            print(f"NEXT: {block.target_file} [{len(block.data)}]")
            if next_b[1] != block.md5():
                print("Block Mismatch!")
            block.unload()
            next_b = block.next
        print(f"LAST_NEXT: {next_b}")

    def _writer(self, block_size):
        exists_blocks = len(self.blocks)
        print(f"Found {exists_blocks} existing blocks.")
        assert exists_blocks > 0, "Root Block Missing"
        block = self.blocks.pop(-1)
        if not block.loaded:
            block.load()

        b = 0
        sample = "placeholder"
        while sample is not None:
            print(f"Adding Block: {exists_blocks + b}")
            block_data = []

            for _i in range(block_size):
                sample = yield
                if sample is None:
                    break
                block_data.append(sample)

            if block_data:
                next_block = Block(self.root, f"block_{exists_blocks + b}_{block_size}.pkl")
                next_block.data = block_data
                next_block.checksum = next_block.md5()
                block.next = [next_block.target_file, next_block.checksum]
            else:
                next_block = None

            block.save()
            # fix mem leakage
            block.unload()
            self.blocks.append(block)
            block = next_block

            b += 1

        if block is not None:
            self.blocks.append(block)
            block.save()

    @contextlib.contextmanager
    def writer(self, block_size=10000):
        writer = self._writer(block_size)
        writer.send(None)
        try:
            yield writer.send
        finally:
            try:
                writer.send(None)
            except StopIteration:
                writer = None
                print("Seal Success.")
            except Exception as e:
                raise e

    def __repr__(self):
        location = (
            f"Chain of {len(self.blocks)} blocks:\n"
            f" > root        = {self.root}\n"
            f" > block id    = {self.block_id}\n"
            f" > block file  = {self.blocks[self.block_id].target_path}\n"
            f" > in-block id = {self.in_block_id}"
        )
        return location

    def state_dict(self):
        return dict(
            root=self.root,
            blocks=[block.state_dict() for block in self.blocks],
            block_id=self.block_id,
            in_block_id=self.in_block_id,
        )

    def load_state_dict(self, state_dict):
        if "root" in state_dict:
            self.root = state_dict["root"]
        self.blocks = (
            [Block(**b) for b in state_dict["blocks"]]
            if len(state_dict["blocks"]) > 0
            else [Block(self.root, "root.pkl")]
        )
        self.root = self.root
        self.block_id = state_dict["block_id"]
        self.in_block_id = state_dict["in_block_id"]
