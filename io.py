import zarr
import pickle
import os
import numpy as np
import sparse
import scipy.sparse as ss
from .csr import csr
from .csc import csc

def save(matrix,store):
    """This function saves zsparse, scipy, and sparse objects."""

    def save_COO(array,store):
        attrs = array.__dict__.copy()
        zarr.array(array.coords,chunks=None,store=store.__class__(os.path.join(store.path,'coords.zarr')))
        del attrs['coords']
        if isinstance(array.data,np.ndarray):
            zarr.array(array.data,chunks=None,store=store.__class__(os.path.join(store.path,'data.zarr')))
            del attrs['data']
        
        with open(os.path.join(store.path,'attrs.pkl'), 'wb') as f:
            pickle.dump(attrs, f, pickle.HIGHEST_PROTOCOL)

    if isinstance(store,str):
        store = zarr.DirectoryStore(store)
    if isinstance(matrix,sparse.COO):
        return save_COO(matrix,store)
    if isinstance(matrix,ss.spmatrix):
        if matrix.format=='csr':
            return csr(matrix,store=store)
        elif matrix.format=='csc':
            return csc(matrix,store=store)
        else:
            raise ValueError('scipy sparse matrices of type {} are not supported'.format(matrix.__class__))
    data_store = store.__class__(os.path.join(store.path,'data.zarr'))
    indices_store = store.__class__(os.path.join(store.path,'indices.zarr'))
    indptr_store = store.__class__(os.path.join(store.path,'indptr.zarr'))
    zarr.array(matrix.data,store=data_store)
    zarr.array(matrix.indices,store=indices_store)
    zarr.array(matrix.indptr,store=indptr_store)
    matrix._store = store
    matrix.persistent = True
    with open(os.path.join(store.path,'attrs.pkl'), 'wb') as f:
        pickle.dump(matrix, f, pickle.HIGHEST_PROTOCOL)
        

def load(filename):
    """A function for loading saved objects. The function behaves differently for zsparse objects
    and for pydata/sparse objects. For the former it none of the array data is loaded into memory.
    pydata/sparse arrays are loaded completely"""
    
    def load_COO(filename):
        
        with open(os.path.join(filename,'attrs.pkl'), 'rb') as f:
            attrs = pickle.load(f) 
        coords = zarr.load(os.path.join(filename,'coords.zarr'))
        
        if os.path.exists(os.path.join(filename,'data.zarr')):
            data = zarr.load(os.path.join(filename,'data.zarr'))
        else:
            data = attrs['data']
        shape = attrs['shape']
        fill_value = attrs['fill_value']
        cache = attrs['_cache']
        return sparse.COO(coords,data,shape=shape,fill_value=fill_value,cache=cache)
    
    if os.path.exists(os.path.join(filename,'coords.zarr')):
        return load_COO(filename)

    with open(os.path.join(filename,'attrs.pkl'), 'rb') as f:
        return pickle.load(f)