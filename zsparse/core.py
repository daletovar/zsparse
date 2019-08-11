import numpy as np 
import scipy.sparse as ss 
import zarr 
import os
import pickle
from zarr.util import InfoReporter, human_readable_size
from .indexing import getitem

FORMATS = {'coo': ss.coo_matrix,
            'csr': ss.csr_matrix,
            'csc': ss.csc_matrix}

FORMAT_NAMES = {'coo': 'Coordinate Sparse Matrix',
            'csr': 'Compressed Sparse Row Matrix',
            'csc': 'Compressed Sparse Column Matrix'}

class Matrix:

    def __init__(self,
                arg,
                format,
                compressor='default',
                shape=None,
                store=None,
                chunks=None,
                dtype=None):

        if format not in FORMATS:
            raise NotImplementedError('The given format is not supported.')
        if not isinstance(arg, ss.spmatrix):
            try:
                arg = FORMATS[format](arg,shape=shape)
            except:
                raise ValueError('Invalid input')
        

        arg = arg.asformat(format)
        self.shape = arg.shape
        if arg.format == 'coo':
            arg = (arg.data,arg.row,arg.col)
        else:
            arg = (arg.data,arg.indices,arg.indptr)
        
        if store is not None:
            store1 = store.__class__(os.path.join(store.path,'data.zarr'))
            if format == 'coo':
                store2 = store.__class__(os.path.join(store.path,'row.zarr'))
                store3 = store.__class__(os.path.join(store.path,'col.zarr'))
            else:
                store2 = store.__class__(os.path.join(store.path,'indices.zarr'))
                store3 = store.__class__(os.path.join(store.path,'indptr.zarr'))
        else:
            store1 = store2 = store3 = None    
        if format == 'coo':
            self.row = zarr.array(arg[1],chunks=chunks,store=store2,compressor=compressor)
            self.col = zarr.array(arg[2],chunks=chunks,store=store3,compressor=compressor)
        else:
            self.indices = zarr.array(arg[1],chunks=chunks,store=store2,compressor=compressor)
            self.indptr = zarr.array(arg[2],chunks=chunks,store=store3,compressor=compressor)
        self.data = zarr.array(arg[0],chunks=chunks,store=store1,compressor=compressor,dtype=dtype)
        self.dtype = self.data.dtype
        self.format = format
        self._store = store
        self._chunk_store = self.data._chunk_store
        self._info_reporter = InfoReporter(self)

        if self._store is not None:
            with open(os.path.join(store.path,'attrs.pkl'), 'wb') as file:
                pickle.dump(self, file)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['data']
        if self.format == 'coo':
            del state['row']
            del state['col']
        else:
            del state['indices']
            del state['indptr']
        return state
    
    def __setstate__(self,state):
        self.__dict__.update(state)
        path = self._store.path
        self.data = zarr.open(os.path.join(path,'data.zarr'))
        if self.format == 'coo':
            self.row = zarr.open(os.path.join(path,'row.zarr'))
            self.col = zarr.open(os.path.join(path,'col.zarr'))
        else:
            self.indices = zarr.open(os.path.join(path,'indices.zarr'))
            self.indptr = zarr.open(os.path.join(path,'indptr.zarr'))

    __getitem__ = getitem

    def __str__(self):
        nbytes = human_readable_size(self.nbytes_stored)
        return "<{}, shape={}, nnz={}, bytes_stored = {}>".format(
            FORMAT_NAMES[self.format],self.shape,self.nnz,nbytes)

    __repr__ = __str__

    @property
    def nchunks(self):
        if self.format == 'coo':
            return self.data.nchunks + self.row.nchunks + self.col.nchunks
        else: 
            return self.data.nchunks + self.indices.nchunks + self.indptr.nchunks
    @property
    def nchunks_initialized(self):
        if self.format == 'coo':
            return self.data.nchunks_initialized + self.row.nchunks_initialized + self.col.nchunks_initialized
        else: 
            return self.data.nchunks_initialized + self.indices.nchunks_initialized + self.indptr.nchunks_initialized
    @property
    def nbytes(self):
        if self.format == 'coo':
            return self.data.nbytes + self.row.nbytes + self.col.nbytes
        else: 
            return self.data.nbytes + self.indices.nbytes + self.indptr.nbytes
    @property
    def nbytes_stored(self):
        if self.format == 'coo':
            return self.data.nbytes_stored + self.row.nbytes_stored + self.col.nbytes_stored
        else: 
            return self.data.nbytes_stored + self.indices.nbytes_stored + self.indptr.nbytes_stored
    @property
    def nnz(self):
        return self.data.shape[0]
    @property
    def density(self):
        return self.nnz/(self.shape[0] * self.shape[1])
    @property
    def compressor(self):
        return self.data.compressor
    @property
    def info(self):
        return self._info_reporter
    def info_items(self):
        return self._info_items_nosync()
    def _info_items_nosync(self):
        def typestr(o):
            return '%s.%s' % (type(o).__module__, type(o).__name__)
        def bytestr(n):
            if n > 2**10:
                return '%s (%s)' % (n, human_readable_size(n))
            else:
                return str(n)
        items = []

        # basic info
        #if self.name is not None:
        #    items += [('Name', self.name)]
        items += [
            ('Type', typestr(self)),
            ('Format', FORMAT_NAMES[self.format]),
            ('Data type', '%s' % self.dtype),
            ('Shape', str(self.shape)),
            ('nnz', '%s' % self.nnz),
            ('Density', str(self.density)),
            ('Order', self.data.order)
        ]

        # compressor
        items += [('Compressor', repr(self.compressor))]
        
        # storage info
        if self._store is not None:
            items += [('Store type', typestr(self._store))]
        if self._chunk_store is not None:
            items += [('Chunk store type', typestr(self._chunk_store))]
        items += [('No. bytes as dense', 
            bytestr(self.shape[0] * self.shape[1] * self.dtype.itemsize))]
        items += [('No. bytes', bytestr(self.nbytes))]
        if self.nbytes_stored > 0:
            items += [
                ('No. bytes stored', bytestr(self.nbytes_stored)),
                ('Storage ratio', '%.1f' % (self.nbytes / self.nbytes_stored)),
            ]
        items += [
            ('Chunks initialized', '%s/%s' % (self.nchunks_initialized, self.nchunks))
        ]

        return items