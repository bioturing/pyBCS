import scipy.sparse
from h5py import _conv, defs, _errors, h5ac, h5a, h5, h5ds, h5f, h5fd, h5g, h5i, h5l, h5o, h5p, h5pl, h5r, h5s, h5t, h5z, _objects, _proxy, utils
import h5py
import numpy as np
import time
import uuid
import os
import sys

def get_partial(n, partial):
    return np.max([1, n // partial])

def replace_dataset(f, name, *args, **kwargs):
    if name in f:
        print("\"%s\" has already existed, overwriting" % name)
        del f[name]
    f.create_dataset(name, *args, **kwargs)

def light_transpose(bcs_path, partial):
    partial = int(partial)
    tmp_path = os.path.join(os.path.dirname(bcs_path), "." + str(uuid.uuid4()))
    print(tmp_path)
    stamp = time.time()
    with h5py.File(bcs_path, "r") as f:
        if "countsT" in f:
            data_slot = "countsT/data"
            if "indices" in f["countsT"]:
                print("Raw data indices are different from normalized data indices")
                indices_slot = "countsT/indices"
                indptr_slot = "countsT/indptr"
            else:
                print("Raw data and normalized data have the same indices")
                indices_slot = "normalizedT/indices"
                indptr_slot = "normalizedT/indptr"
        else:
            data_slot = "normalizedT/data"
            indices_slot = "normalizedT/indices"
            indptr_slot = "normalizedT/indptr"

    with h5py.File(bcs_path, "r") as f:
        nnz = len(f[indices_slot])
        with h5py.File(tmp_path, "w") as g:
            n, m = f["normalizedT"]["shape"][:]
            k = get_partial(m, partial)
            i = 0
            while i < m:
                pre = time.time()
                p = np.min([k, m - i])
                indptr = f[indptr_slot][:]
                l, r = indptr[i], indptr[i + p]
                indptr[0:i] = 0
                indptr[i:] -= indptr[i]
                indptr[i + p:] = indptr[i + p]
                mat = scipy.sparse.csc_matrix((f[data_slot][l:r], f[indices_slot][l:r], indptr),
                                                shape=[n, m])
                mat = mat.transpose().tocsc()
                group = g.create_group(str(i))
                group.create_dataset("indptr", data=mat.indptr)
                group.create_dataset("indices", data=mat.indices)
                group.create_dataset("data", data=mat.data)
                i += p
                print("Done a circle in %f seconds (%.2f%%)" % (time.time() - pre, i / m * 100), flush=True)

    print("Done splitting matrix in %f seconds" % (time.time() - stamp))

    stamp = time.time()
    with h5py.File(tmp_path, "r") as f:
        with h5py.File(bcs_path, "a") as g:
            replace_dataset(g, name="bioturing/data", shape=(nnz, ), dtype="f4")
            replace_dataset(g, name="bioturing/indices", shape=(nnz, ), dtype="i4")
            joined_indptr = []
            k = get_partial(n, partial)
            i = 0
            ptr = 0
            while i < n:
                pre = time.time()
                p = np.min([k, n - i])
                matrices = []
                for j in range(0, m, get_partial(m, partial)):
                    indptr = f[str(j)]["indptr"][:]
                    l, r = indptr[i], indptr[i + p]
                    indptr[0:i] = 0
                    indptr[i:] -= indptr[i]
                    indptr[i + p:] = indptr[i + p]
                    mat = scipy.sparse.csc_matrix((f[str(j)]["data"][l:r], f[str(j)]["indices"][l:r] - j, indptr),
                                                    shape=[np.min([get_partial(m, partial), m - j]), n])
                    matrices.append(mat)

                joined = scipy.sparse.vstack(matrices, format="csc")
                #joined = np.sum(matrices)
                del matrices
                size = len(joined.data)
                g["bioturing/data"][ptr:ptr + size] = joined.data
                g["bioturing/indices"][ptr:ptr + size] = joined.indices
                joined_indptr.extend(joined.indptr[i:i+p] + ptr)
                ptr += size
                i += p
                print("Done a circle in %f seconds (%.2f%%)" % (time.time() - pre, i / n * 100), flush=True)
            joined_indptr.append(nnz)
            replace_dataset(g, name="bioturing/indptr", data=joined_indptr)
            replace_dataset(g, name="bioturing/shape", data=[m, n])
            replace_dataset(g, name="bioturing/barcodes", data=g["normalizedT"]["features"])
            replace_dataset(g, name="bioturing/features", data=g["normalizedT"]["barcodes"])
            replace_dataset(g, name="bioturing/feature_type", data=["RNA".encode("utf8")] * m)
    print("Done joining matrices in %f seconds" % (time.time() - stamp), flush=True)
    os.remove(tmp_path)

def create_countsT(bcs_path):
    print("Creating \"countsT\"")
    with h5py.File(bcs_path, "a") as f:
        if "countsT" not in f:
            print("Raw data is not available, ignoring \"countsT\"")
            return
        if "indptr" not in f["countsT"]:
            replace_dataset(f, name="countsT/indptr", data=f["normalizedT/indptr"])
        if "indices" not in f["countsT"]:
            replace_dataset(f, name="countsT/indices", data=f["normalizedT/indices"])
        replace_dataset(f, name="countsT/shape", data=f["normalizedT/shape"])
        replace_dataset(f, name="countsT/features", data=f["normalizedT/features"])
        replace_dataset(f, name="countsT/barcodes", data=f["normalizedT/barcodes"])

def bcs_transpose(bcs_path, partial):
    light_transpose(bcs_path, partial)
    create_countsT(bcs_path)

if __name__ == "__main__":
    bcs_transpose(sys.argv[1], sys.argv[2])

