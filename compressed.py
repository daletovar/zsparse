import scipy.sparse as ss
import numpy as np
import zarr
import pickle
import os
from zarr.util import InfoReporter, human_readable_size
from warnings import warn
import scipy.sparse as ss

FORMATS = {'csr':'Compressed Sparse Row Matrix',
            'csc':'Compressed Sparse Column Matrix'}    

class cs_matrix():
    
    def __init__(self,arg,shape=None,dtype=None,
                 data_chunks=None,indptr_chunks=None,store=None,scipy=False,
                 check_format=False):
                
                
        if isinstance(arg,tuple):
            data,indices,indptr = arg[0],arg[1],arg[2]
            if not len(indices)==len(data) and indptr[-1]==len(indices):
                raise ValueError('matrix of type {} is not in canonical format.'.format(self.__class__))
        
        elif isinstance(arg,np.ndarray) and arg.ndim == 2:
            data, indices, indptr, shape = self._from_dense(arg)
        
        
        elif isinstance(arg,ss.spmatrix):
            data,indices,indptr,shape = arg.data,arg.indices,arg.indptr,arg.shape
        else:
            raise ValueError('Invalid input')

        if store is not None:
            data_store = store.__class__(os.path.join(store.path,'data.zarr'))
            indices_store = store.__class__(os.path.join(store.path,'indices.zarr'))
            indptr_store = store.__class__(os.path.join(store.path,'indptr.zarr'))
        else:
            data_store = None
            indices_store = None
            indptr_store = None

        #to save time
        if not all(isinstance(i, zarr.Array) for i in [data, indices, indices]):
            self.data = zarr.array(data,chunks=data_chunks,dtype=dtype,store=data_store)
            self.indices = zarr.array(indices,chunks=data_chunks,store=indices_store)
            self.indptr = zarr.array(indptr,chunks=indptr_chunks,store=indptr_store)
        else:
            self.data = data
            self.indices = indices
            self.indptr = indptr

            if not data.dtype==dtype:
                warn('Given dtype {dtype} does not agree with dtype of the data entered {data.dtype}.')
            if self.indptr.dtype.kind != 'i':
                warn("indptr array has non-integer dtype ({})"
                 "".format(self.indptr.dtype.name), stacklevel=3)
            if self.indices.dtype.kind != 'i':
                warn("indices array has non-integer dtype ({})"
                "".format(self.indices.dtype.name), stacklevel=3)
            
            
        
        
        if shape is not None:
            self.shape = shape
        else:
            self.shape = (len(indptr)-1,np.max(indices))
        
        
        
        
        # initialize info reporter
        self._info_reporter = InfoReporter(self)
        self.nnz = self.data.shape[0]
        self.density = self.nnz/(self.shape[0]*self.shape[1])
        self.data_compressor = self.data.compressor
        self.indices_compressor = self.indices.compressor
        self.indptr_compressor = self.indptr.compressor
        self.scipy = scipy
        self._store = store
        self.name = None if self._store is None else self._store.path
        self.persistent = False if self._store is None else True
        self._chunk_store = self.data._chunk_store
        self.nbytes = self.data.nbytes + self.indices.nbytes + self.indptr.nbytes
        self.nbytes_stored = self.data.nbytes_stored + self.indices.nbytes_stored + self.indptr.nbytes_stored
        self.dtype = self.data.dtype
        self.nchunks = self.data.nchunks + self.indices.nchunks + self.indptr.nchunks
        self.nchunks_initialized = self.data.nchunks_initialized + self.indices.nchunks_initialized + self.indptr.nchunks_initialized
        self.read_only = True
        
        if self.persistent==True:
            with open(os.path.join(self._store.path,'attrs.pkl'), 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


    def __repr__(self):
        format_name = FORMATS[self.format]
        nbytes = human_readable_size(self.nbytes_stored)
        return "<{}, shape={}, nnz={}, bytes_stored = {}>".format(format_name,self.shape,self.nnz,nbytes)


    def __getstate__(self):
        state = self.__dict__.copy()
        #del state['root']
        del state['data']
        del state['indices']
        del state['indptr']
        return state
    def __setstate__(self,state):
        self.__dict__.update(state)
        path = self._store.path
        self.data = zarr.open(os.path.join(path,'data.zarr'))
        self.indices = zarr.open(os.path.join(path,'indices.zarr'))
        self.indptr = zarr.open(os.path.join(path,'indptr.zarr'))
        
    def update_info(self):
        
        # initialize info reporter
        self._info_reporter = InfoReporter(self)
        self.nnz = self.data.shape[0]
        self.data_compressor = self.data.compressor
        self.indices_compressor = self.indices.compressor
        self.indptr_compressor = self.indptr.compressor
        #self._store = store
        self.persistent = False if self._store is None else True
        self._chunk_store = self.data._chunk_store
        self.nbytes = self.data.nbytes + self.indices.nbytes + self.indptr.nbytes
        self.nbytes_stored = self.data.nbytes_stored + self.indices.nbytes_stored + self.indptr.nbytes_stored
        self.dtype = self.data.dtype
        self.nchunks = self.data.nchunks + self.indices.nchunks + self.indptr.nchunks
        self.nchunks_initialized = self.data.nchunks_initialized + self.indices.nchunks_initialized + self.indptr.nchunks_initialized
        self.read_only = True
    
    
    
    
    def todense(self):
        if self.format=='csr':
            return ss.csr_matrix((self.data[:],self.indices[:],self.indptr[:]),shape=self.shape).toarray()
        else:
            return ss.csc_matrix((self.data[:],self.indices[:],self.indptr[:]),shape=self.shape).toarray()

    def to_scipy(self):
        if self.format=='csr':
            return ss.csr_matrix((self.data[:],self.indices[:],self.indptr[:]),shape=self.shape)
        else:
            return ss.csc_matrix((self.data[:],self.indices[:],self.indptr[:]),shape=self.shape)


    @property
    def T(self):
        return self.transpose()

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
        if self.name is not None:
            items += [('Name', self.name)]
        items += [
            ('Type', typestr(self)),
            ('Format', 'Compressed Sparse Row Matrix'),
            ('Data type', '%s' % self.dtype),
            ('Shape', str(self.shape)),
            ('nnz', '%s' % self.nnz),
            ('Density', str(self.density)),
            ('Order', self.data.order),
            ('Read-only', str(self.read_only)),
            ('Persistent', str(self.persistent))
        ]

        # compressor
        items += [('Data compressor', repr(self.data_compressor))]
        items += [('Indices compressor', repr(self.indices_compressor))]
        items += [('Indptr compressor', repr(self.indptr_compressor))]
        
        # storage info
        if self._store is not None:
            items += [('Store type', typestr(self._store))]
        if self._chunk_store is not None:
            items += [('Chunk store type', typestr(self._chunk_store))]
        items += [('No. bytes as dense', bytestr(self.shape[0]*self.shape[1]*self.dtype.itemsize))]
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
        
    
    def _from_dense(self,mat):
        if self.format == 'csr':
            mat = ss.csr_matrix(mat)
        else:
            mat = ss.csc_matrix(mat)
        
        return mat.data,mat.indices,mat.indptr,mat.shape
        
        
    def __eq__(self,other):
        
        if isinstance(other,self.__class__):
            if self.data == other.data and self.indices==other.indices and self.indptr==other.indptr:
                return True
            
        elif isinstance(other,np.ndarray):
            if self.shape == other.shape:
                return self.to_scipy()==other
        return False

        
    def __ne__(self,other):
        if isinstance(other,self.__class__):
            if self.data == other.data and self.indices==other.indices and self.indptr==other.indptr:
                return False
        elif isinstance(other,np.ndarray):
            if self.shape == other.shape:
                return self.to_scipy()!=other
        return True
        
        
        
        
    