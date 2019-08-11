import pickle
import zarr
import os
import sparse
import numpy as np 
import scipy.sparse as ss 
from .core import Matrix
from sparse import SparseArray


FORMATS = {'coo': ss.coo_matrix,
            'csr': ss.csr_matrix,
            'csc': ss.csc_matrix}

def save(mat, store, compressor='default'):
    """ 
    Save a sparse matrix to disk in numpy's ``.npz`` format.
    Note: This is not binary compatible with scipy's ``save_npz()``.
    Will save a file that can only be opend with this package's ``load_npz()``.
    Parameters
    ----------
    filename : string or file
        Either the file name (string) or an open file (file-like object)
        where the data will be saved. If file is a string or a Path, the
        ``.npz`` extension will be appended to the file name if it is not
        already there
    matrix : COO
        The matrix to save to disk
    compressed : bool
        Whether to save in compressed or uncompressed mode
    Example
    --------
    Store sparse matrix to disk, and load it again:
    >>> import os
    >>> import sparse
    >>> import numpy as np
    >>> dense_mat = np.array([[[0., 0.], [0., 0.70677779]], [[0., 0.], [0., 0.86522495]]])
    >>> mat = sparse.COO(dense_mat)
    >>> mat
    <COO: shape=(2, 2, 2), dtype=float64, nnz=2, fill_value=0.0>
    >>> sparse.save_npz('mat.npz', mat)
    >>> loaded_mat = sparse.load_npz('mat.npz')
    >>> loaded_mat
    <COO: shape=(2, 2, 2), dtype=float64, nnz=2, fill_value=0.0>
    >>> os.remove('mat.npz')
    See Also
    --------
    load_npz
    scipy.sparse.save_npz
    scipy.sparse.load_npz
    numpy.savez
    numpy.load
    """

    if isinstance(store,str):
        store = zarr.DirectoryStore(store)
    
    if isinstance(mat,SparseArray):
        return _save_array(mat, store, compressor)
    
    if not mat.format in FORMATS:
        raise NotImplementedError('cannot save matrix of format {}'.format(
            mat.format))
    if isinstance(mat,ss.spmatrix):
        return Matrix(mat,mat.format,store=store,compressor=compressor)

    store1 = type(store)(os.path.join(store.path,'data.zarr'))
    if format == 'coo':
        store2 = type(store)(os.path.join(store.path,'row.zarr'))
        store3 = type(store)(os.path.join(store.path,'col.zarr'))
    else:
        store2 = type(store)(os.path.join(store.path,'indices.zarr'))
        store3 = type(store)(os.path.join(store.path,'indptr.zarr'))

    if isinstance(mat,Matrix):
        zarr.array(mat.data,store=store1,compressor=compressor)
        if mat.format == 'coo':
            zarr.array(mat.row,store=store2,compressor=compressor)
            zarr.array(mat.col,store=store3,compressor=compressor) 
        else:
            zarr.array(mat.indices,store=store2,compressor=compressor)
            zarr.array(mat.indptr,store=store3,compressor=compressor)
        mat._store = store   
    
    with open(os.path.join(store.path,'attrs.pkl'), 'wb') as file:
        pickle.dump(mat, file)


def _save_array(arr, store,compressor):
    attrs = arr.__dict__.copy()
    if isinstance(arr.data, np.ndarray):
        zarr.array(arr.data,store=type(store)(os.path.join(store.path,'data.zarr')),
            compressor=compressor)

    if isinstance(arr, sparse.COO):
        zarr.array(arr.coords, store=type(store)(os.path.join(store.path,'coords.zarr')),
            compressor=compressor)
        del attrs['coords']
    elif isinstance(arr, sparse.GXCS):
        zarr.array(arr.indices,store=type(store)(os.path.join(store.path,'indices.zarr')),
            compressor=compressor)
        zarr.array(arr.indptr,store=type(store)(os.path.join(store.path,'indptr.zarr')),
            compressor=compressor)
        del attrs['indices']
        del attrs['indptr']
    else:
        raise NotImplementedError('`{}` format is not supported'.format(type(arr)))
    
    with open(os.path.join(store.path,'attrs.pkl'), 'wb') as file:
            pickle.dump(attrs, file)

def load(filename):

    if os.path.exists(filename):
        try:
            with open(os.path.join(filename,'attrs.pkl'), 'rb') as file:
                arr = pickle.load(file)
        except:
            raise ValueError('a sparse matrix/array does not exist at this location')
        
        if isinstance(arr, Matrix):
            return arr
        else:
            shape = arr['shape']
            fill_value = arr['fill_value']
            if '_cache' in arr: cache = arr['_cache']
            if 'compressed_axes' in arr: cp_ax = arr['compressed_axes']
            if 'data' in arr: data = arr['data']
            if os.path.exists(os.path.join(filename,'data.zarr')):
                data = zarr.load(os.path.join(filename,'data.zarr'))
            if os.path.exists(os.path.join(filename,'coords.zarr')):
                coords = zarr.load(os.path.join(filename,'coords.zarr'))
                return sparse.COO(coords,
                    data,shape=shape,fill_value=fill_value,cache=cache)
            elif os.path.exists(os.path.join(filename,'indptr.zarr')):
                indptr = zarr.load(os.path.join(filename,'indptr.zarr'))
                indices = zarr.load(os.path.join(filename,'indices.zarr'))
                return sparse.GXCS((data,indices,indptr),shape=shape,
                    compressed_axes=cp_ax,fill_value=fill_value)

