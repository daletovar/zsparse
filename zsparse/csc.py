from zarr.util import InfoReporter, human_readable_size
import scipy.sparse as ss
import numpy as np
from .indexing import IndexMixin
from .compressed import cs_matrix

class csc(cs_matrix,IndexMixin):
    
    format = 'csc'

    def append(self,to_append):
        if isinstance(to_append,csc) or isinstance(to_append,ss.csc_matrix):
            assert to_append.shape[0]==self.shape[0]
                
        elif hasattr(to_append,'shape'):
            assert to_append.ndim == 2
            to_append = ss.csc_matrix(to_append)
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
        from .csr import csr

        return csr((self.data, self.indices,
                           self.indptr), shape=(N, M))


    def _get_element(self,row,col): 
        s = np.searchsorted(self.indices[self.indptr[col]:self.indptr[col+1]],row)
        if self.indices[s] == row:
	        return self.data[s]
        else:
	        return 0
    
    def _get_col(self,row,col): 
        data = self.data[self.indptr[col]:self.indptr[col+1]]
        indices = self.indices[self.indptr[col]:self.indptr[col+1]]
        indptr = np.array([0,indices.shape[0]])
        shape = (self.shape[0], 1)
        return csc((data,indices,indptr),shape=shape)
        
    def _get_col_slice(self,row,col): 

        if col.step is not None:
	        raise NotImplementedError("Index step is not supported.")
        indptr_slice = slice(col.start, col.stop)
        indptr = self.indptr[indptr_slice]
        data = self.data[indptr[0]:indptr[-1]]
        indices = self.indices[indptr[0]:indptr[-1]]
        indptr -= indptr[0]
        shape = (self.shape[0], indptr.size - 1)
        return csc((data,indices,indptr),shape=shape)
	
	
    def _get_col_fancy(self,row,col): 
        
        coords1 = self.indptr.get_coordinate_selection(col)
        coords2 = self.indptr.get_coordinate_selection(col+1)
        indptr = np.empty((len(col)+1))
        indptr[0] = 0
        ind_list = []
        for i in range(len(col)):
            inds = np.arange(coords1[i],coords2[i])
            ind_list.extend(inds)
            indptr[i+1] = indptr[i] + inds.shape[0]
        data = self.data.get_coordinate_selection(ind_list)
        indices = self.indices.get_coordinate_selection(ind_list)
        return csc((data,indices,indptr),shape=(self.shape[0],len(col)))
	
    def _get_partial_col(self,row,col): 
        """X[5:10,5]"""
        col = np.array([col])
        return self._row_slicing(row,col)

    def _row_int(self,row,col): 
        """handles cases of indexing rows with ints"""
        inds = []
        indptr = np.empty(len(col)+1)
        coords1 = self.indptr.get_coordinate_selection(col)
        coords2 = self.indptr.get_coordinate_selection(col+1)
        for i in range(len(col)):
            s = np.searchsorted(self.indices[coords1[i]:coords2[i]],row) + coords1[i]
            if self.indices[s]==row:
                inds.append(s)
                indptr[i+1] = indptr[i]+1
            else:
                indptr[i+1] = indptr[i]
        shape = (1,indptr.shape[0]-1)
        if inds == []:
            return csc(([],[],indptr),shape=shape)
        indices = self.indices.get_coordinate_selection(inds) - row
        data = self.data.get_coordinate_selection(inds)
        return csc((data,indices,indptr),shape=shape)

    def _get_row(self,row,col):
        """X[10,:]"""
        col = np.arange(len(self.indptr)-1)
        return self._row_int(row,col)
    
    def _get_partial_row(self,row,col):
        """X[9,5:20]"""
        col = np.arange(col.start,col.stop)
        return self._row_int(row,col)

    def _row_slicing(self,row,col):
        """This function handles all cases of row slicing"""

        inds_list = []
        indptr = np.empty(len(col)+1)
        indptr[0] = 0
        coords1 = self.indptr.get_coordinate_selection(col)
        coords2 = self.indptr.get_coordinate_selection(col+1)
        for i in range(len(col)):
            start = np.searchsorted(self.indices[coords1[i]:coords2[i]], row.start) + coords1[i]
            stop = np.searchsorted(self.indices[coords1[i]:coords2[i]], row.stop) + coords1[i]
            inds = np.arange(start,stop)
            inds_list.extend(inds)
            indptr[i+1] = indptr[i] + inds.size
        shape = (row.stop - row.start,len(col))
        if inds_list == []:
            return csc(([],[],indptr),shape=shape)
        indices = self.indices.get_coordinate_selection(inds_list) - row.start
        data = self.data.get_coordinate_selection(inds_list)
        return csc((data,indices,indptr),shape=shape)       

    def _get_row_slice(self,row,col):
        """X[5:60,:]"""
        col = np.arange(len(self.indptr)-1)
        return self._row_slicing(row,col)
    
    def _get_slices(self,row,col):
        """X[5:50,10:200]"""
        col = np.arange(col.start,col.stop)
        return self._row_slicing(row,col)
    def _row_slice_col_fancy(self,row,col):
        """X[100:500,[2,5,88]]"""
        return self._row_slicing(row,col)

    def _row_fancy_col_slice(self,row,col):
        raise NotImplementedError('Indexing with a {} in the first dimension of class {} is not supported'.format(type(row),self.__class__))
