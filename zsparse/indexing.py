import numpy as np 
import scipy.sparse as ss 
from numbers import Integral
from collections.abc import Iterable
from .slicing import normalize_index

FORMATS = {'csr': ss.csr_matrix,
            'csc': ss.csc_matrix}

def getitem(x, key):

    if x.format == 'coo':
        raise NotImplementedError('indexing with coo matrices is not supported.')
    
    key = normalize_index(key, x.shape)
    shape = []

    for k in key:
        if isinstance(k,Integral):
            shape.append(1)
        elif isinstance(k,slice):
            shape.append(len(range(k.start, k.stop, k.step)))
        elif isinstance(k,np.ndarray):
            shape.append(k.size)
    
    shape = tuple(shape)
    
    if x.format == 'csc':
        key = [key[1], key[0]]
    compressed_idx, uncompressed_idx = key

    # single element indexing
    if all(isinstance(ind,Integral) for ind in (compressed_idx,uncompressed_idx)):
        return find_single_element(x,compressed_idx,uncompressed_idx)

    # convert to ndarray
    if isinstance(compressed_idx,Integral):
        compressed_idx = [compressed_idx]
    if not isinstance(compressed_idx, np.ndarray):
        if isinstance(compressed_idx, slice):
            compressed_idx = np.array(range(
                compressed_idx.start,compressed_idx.stop,compressed_idx.step))
        elif isinstance(compressed_idx, Iterable):
            compressed_idx = np.array(compressed_idx)
        else:
            raise IndexError('invalid input for index along compressed dimension')
            
    
    if isinstance(uncompressed_idx, slice):
        if uncompressed_idx.step != 1:
            uncompressed_idx = np.array(range(
                uncompressed_idx.start,uncompressed_idx.stop,uncompressed_idx.step))
    elif isinstance(uncompressed_idx,Integral):
        uncompressed_idx = [uncompressed_idx]
    
    # optimized slicing
    if isinstance(uncompressed_idx,slice):
        arg = uncompressed_slicing(x,compressed_idx,uncompressed_idx)
    else:
        arg = uncompressed_fancy(x,compressed_idx,uncompressed_idx)
    
    return FORMATS[x.format](arg,shape=shape)

def find_single_element(x, compressed_idx, uncompressed_idx):
    item = np.searchsorted(x.indices[x.indptr[compressed_idx:compressed_idx + 1]],
        uncompressed_idx) + x.indptr[compressed_idx]
    if x.indices[item] == uncompressed_idx:
        return x.data[item]
    return 0 

def uncompressed_slicing(x, compressed_idx, uncompressed_idx):
    nnz = 0
    starts = x.indptr.get_coordinate_selection(compressed_idx)
    stops = x.indptr.get_coordinate_selection(compressed_idx + 1)
    #stops = x.indptr[slice(compressed_idx.start + 1, compressed_idx.stop + 1, 1)]
    indptr = np.empty(starts.size + 1, dtype=np.intp)
    indptr[0] = 0 
    begins = []
    ends = []
    for i,(start,stop) in enumerate(zip(starts,stops),1):
        current_row = x.indices[start:stop]
        if len(current_row) == 0: continue
        begins.append(np.searchsorted(current_row,uncompressed_idx.start) + start)
        ends.append(np.searchsorted(current_row, uncompressed_idx.stop) + start)
        nnz += ends[-1] - begins[-1]
        indptr[i] = nnz
    data = np.empty(nnz,dtype=x.data.dtype)
    indices = np.empty(nnz,dtype=np.intp)
    
    # this might be slower
    for i, (begin, end) in enumerate(zip(begins,ends)):
        data[indptr[i]:indptr[i+1]] = x.data[begin:end]
        indices[indptr[i]:indptr[i+1]] = x.indices[begin:end]
    return (data,indices,indptr)

def uncompressed_fancy(x, compressed_idx, uncompressed_idx):
    starts = x.indptr.get_coordinate_selection(compressed_idx)
    stops = x.indptr.get_coordinate_selection(compressed_idx + 1)
    indptr = np.empty(starts.size + 1, dtype=np.intp)
    indptr[0] = 0 
    indices = []
    ind_list = []
    for i,(start,stop) in enumerate(zip(starts,stops)):
        inds = []
        current_row = x.indices[start:stop]
        for u in range(len(uncompressed_idx)):
            s = np.searchsorted(current_row, uncompressed_idx[u])
            if not (s >= current_row.size or current_row[s] != uncompressed_idx[u]):
                s += start
                inds.append(s)
                indices.append(u)
        ind_list.extend(inds)
        indptr[i + 1] = indptr[i] + len(inds)
    ind_list = np.array(ind_list, dtype=np.int64)
    indices = np.array(indices)
    data = x.data.get_coordinate_selection(ind_list)
    return (data,indices,indptr)
             

    