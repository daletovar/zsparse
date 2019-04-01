from zarr.util import InfoReporter, human_readable_size
import scipy.sparse as ss
import numpy as np
from .compressed import cs_matrix
from .indexing import IndexMixin
class csr(cs_matrix,IndexMixin):
    
    format = 'csr'

    def append(self,to_append):
        if isinstance(to_append,csr) or isinstance(to_append,ss.csr_matrix):
            assert to_append.shape[1]==self.shape[1]
                
                
        elif hasattr(to_append,'shape'):
            assert to_append.ndim == 2
            to_append = ss.csr_matrix(to_append)
            assert to_append.shape[1]==self.shape[1]
        else:
            raise TypeError('Object of type {} is not a valid input.'.format(type(to_append)))
            
        self.data.append(to_append.data)
        self.indices.append(to_append.indices)
        self.indptr.append(to_append.indptr + self.indptr[-1])
        self.shape = (self.shape[0] + to_append.shape[0], self.shape[1])
        self.update_info()
            
    def transpose(self,axes=None):
        if axes is not None:
            raise ValueError(("Sparse matrices do not support "
                              "an 'axes' parameter because swapping "
                              "dimensions is the only logical permutation."))
        M, N = self.shape
        from .csc import csc
        return csc((self.data, self.indices,
                           self.indptr), shape=(N, M))

    
    def _get_element(self,row,col):
        s = np.searchsorted(self.indices[self.indptr[row]:self.indptr[row+1]],col)
        if self.indices[s] == col:
	        return self.data[s]
        else:
	        return 0
    
    def _get_row(self,row,col):
        data = self.data[self.indptr[row]:self.indptr[row+1]]
        indices = self.indices[self.indptr[row]:self.indptr[row+1]]
        indptr = np.array([0,indices.shape[0]])
        shape = (1, self.shape[1])
        return csr((data,indices,indptr),shape=shape)
        
    def _get_row_slice(self,row,col):

        if row.step is not None:
	        raise NotImplementedError("Index step is not supported.")
        indptr_slice = slice(row.start, row.stop)
        indptr = self.indptr[indptr_slice]
        data = self.data[indptr[0]:indptr[-1]]
        indices = self.indices[indptr[0]:indptr[-1]]
        indptr -= indptr[0]
        shape = (indptr.size - 1, self.shape[1])
        return csr((data,indices,indptr),shape=shape)
	
	
    def _get_row_fancy(self,row,col):
        
        coords1 = self.indptr.get_coordinate_selection(row)
        coords2 = self.indptr.get_coordinate_selection(row+1)
        indptr = np.empty((len(row)+1))
        indptr[0] = 0
        ind_list = []
        for i in range(len(row)):
            inds = np.arange(coords1[i],coords2[i])
            ind_list.extend(inds)
            indptr[i+1] = indptr[i] + inds.shape[0]
        data = self.data.get_coordinate_selection(ind_list)
        indices = self.indices.get_coordinate_selection(ind_list)
        return csr((data,indices,indptr),shape=(len(row),self.shape[1]))
	
    def _get_partial_row(self,row,col):
        row = np.array([row])
        return self._col_slicing(row,col)

    def _col_int(self,row,col):
        """handles cases of indexing columns with ints"""
        inds = []
        indptr = np.empty(len(row)+1)
        coords1 = self.indptr.get_coordinate_selection(row)
        coords2 = self.indptr.get_coordinate_selection(row+1)
        for i in range(len(row)):
            s = np.searchsorted(self.indices[coords1[i]:coords2[i]],col) + coords1[i]
            if self.indices[s]==col:
                inds.append(s)
                indptr[i+1] = indptr[i]+1
            else:
                indptr[i+1] = indptr[i]
        shape = (indptr.shape[0]-1,1)
        if inds == []:
            return csr(([],[],indptr),shape=shape)
        indices = self.indices.get_coordinate_selection(inds) - col
        data = self.data.get_coordinate_selection(inds)
        return csr((data,indices,indptr),shape=shape)

    def _get_col(self,row,col):
        """X[:,10]"""
        row = np.arange(len(self.indptr)-1)
        return self._col_int(row,col)
    
    def _get_partial_col(self,row,col):
        """X[5:20,9]"""
        row = np.arange(row.start,row.stop)
        return self._col_int(row,col)

    def _col_slicing(self,row,col):
        """This function handles all cases of column slicing"""

        inds_list = []
        indptr = np.empty(len(row)+1)
        indptr[0] = 0
        coords1 = self.indptr.get_coordinate_selection(row)
        coords2 = self.indptr.get_coordinate_selection(row+1)
        for i in range(len(row)):
            start = np.searchsorted(self.indices[coords1[i]:coords2[i]], col.start) + coords1[i]
            stop = np.searchsorted(self.indices[coords1[i]:coords2[i]], col.stop) + coords1[i]
            inds = np.arange(start,stop)
            inds_list.extend(inds)
            indptr[i+1] = indptr[i] + inds.size
        shape = (len(row), col.stop - col.start)
        if inds_list == []:
            return csr(([],[],indptr),shape=shape) 
        indices = self.indices.get_coordinate_selection(inds_list) - col.start
        data = self.data.get_coordinate_selection(inds_list)
        return csr((data,indices,indptr),shape=shape)       

    def _get_col_slice(self,row,col):
        """X[:,5:60]"""
        row = np.arange(len(self.indptr)-1)
        return self._col_slicing(row,col)
    
    def _get_slices(self,row,col):
        """X[5:50,10:200]"""
        row = np.arange(row.start,row.stop)
        return self._col_slicing(row,col)
    
    def _row_fancy_col_slice(self,row,col):
        """X[[2,5,88],100:500]"""
        return self._col_slicing(row,col)

    def _row_slice_col_fancy(self,row,col):
        raise NotImplementedError('Indexing with a {} in the second dimension of class {} is not supported.'.format(type(col),self.__class__))
    
    