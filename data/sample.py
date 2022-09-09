import pickle
import zlib
import lmdb


class PickleDatabase(object):

    def __init__(self, database_path, write=False, map_size=32 * 1024 * 1024 * 1024):
        self._lmdb_env = lmdb.open(database_path, map_size=map_size, lock=write, readonly=(not write))
        self._write = write
        self._id_set = set()
        self._database_path = database_path

    def __del__(self):
        # write back the meta data
        if self._write:
            id_list = []
            for id in self._id_set:
                id_list.append(id)
            self.put_id('id_list', id_list)

            print(f"pickle database [{self._database_path}] meta_data saved!")

        # close dataset
        self._lmdb_env.close()
        # print(f"pickle database [{self._database_path}] closed!")



    def get(self, key: str):
        with self._lmdb_env.begin() as txn:
            decompressor = zlib.decompressobj(wbits=16 + zlib.MAX_WBITS)
            sample_bytes_compressed = txn.get(key.encode('utf-8'))
            if sample_bytes_compressed is None:
                raise Exception(f'{key} not found in database')
            sample_bytes = decompressor.decompress(sample_bytes_compressed)
            sample_bytes += decompressor.flush()
            sample = pickle.loads(sample_bytes)  # sample is object
        return sample

    def put(self, key: str, sample):
        if not self._write:
            return False

        with self._lmdb_env.begin(write=True) as txn:
            compressor = zlib.compressobj(9, wbits=16 + zlib.MAX_WBITS)
            sample_bytes = pickle.dumps(sample)  # sample is object
            sample_bytes_compressed = compressor.compress(sample_bytes)
            sample_bytes_compressed += compressor.flush()
            res = txn.put(key.encode('utf-8'), sample_bytes_compressed)
            self._id_set.add(key)


        return res


    
    def put_id(self, key: str, sample_raw):
        if not self._write:
            return False

        with self._lmdb_env.begin(write=True) as txn:
            # lmdb put must be bytestring
            compressor = zlib.compressobj(9, wbits=16 + zlib.MAX_WBITS)
            sample_bytes = pickle.dumps(sample_raw)  # sample is object
            sample_bytes_compressed = compressor.compress(sample_bytes)
            sample_bytes_compressed += compressor.flush()
            res = txn.put(key.encode('utf-8'), sample_bytes_compressed)
        return res

