import bisect
import numpy as np


INT_TYPES = (int, np.integer)

class IndexMixin:
    
    def __getitem__(self,key):
        
        row,col = self._validate_indices(key) 
        
        #single element
        # m[3,8]
        if all(isinstance(i, INT_TYPES) for i in [row,col]):
            return self._get_element(row,col)
        
        # indexing with two lists or arrays
        if all(hasattr(i, '__len__') for i in [row,col]):
            raise NotImplementedError(('Object of class {} does not support indexing'
                                        'with iterables in both dimensions.'.format(self.__class__)))
        
        
            #csr optimized methods
            # m[5,:]   - self._get_row(row)
            # m[20:1000,:]  - self._get_row_slice(row)
            # m[[0,20,50,88,555],:]
        if isinstance(col,slice) and col==slice(None):
            if isinstance(row,INT_TYPES):
                return self._get_row(row,col)
            elif isinstance(row,slice):
                return self._get_row_slice(row,col)
            else:
                return self._get_row_fancy(row,col)
        
            #csc optimized methods
            # m[:,5]
            # m[:,20:1000]
            # m[:,[0,20,50,88,555]]
        if isinstance(row,slice) and row==slice(None):
            if isinstance(col,INT_TYPES):
                return self._get_col(row,col)
            elif isinstance(col,slice):
                return self._get_col_slice(row,col)
            else:
                return self._get_col_fancy(row,col)
        
        if isinstance(row,slice):
            # m[5:20,45]
            # m[5:20,5:20]
            # m[5:20,[2,8,15,18]]
            if isinstance(col,INT_TYPES):
                return self._get_partial_col(row,col)
            elif isinstance(col,slice):
                #return self._col)sl
                return self._get_slices(row,col)
            else:
                return self._row_slice_col_fancy(row,col)
        
        elif hasattr(row,'__len__'):
            # m[[2,3,4,6,8],8]
            if isinstance(col,INT_TYPES):
                return self._col_int(row,col)
            

        
            
        if isinstance(col,slice): 
            # m[5,5:50]
            # m[[2,3,7,8],5:50]
            if isinstance(row,INT_TYPES):
                return self._get_partial_row(row,col)
            elif hasattr(row,'__len__'):
                return self._row_fancy_col_slice(row,col)
        
        elif hasattr(col,'__len__'):
            # m[8,[2,3,4,6,8]]
            if isinstance(row,INT_TYPES):
                return self._row_int(row,col)
        
        
    def _validate_indices(self,key):
        M,N = self.shape
        row,col = key
        if isinstance(row,INT_TYPES):
            if row < 0 or row >=  M:
                raise IndexError('Index out of bounds')
        elif isinstance(row,slice):
            if not row==slice(None):
                if row.start == None:
                    row = slice(0,row.stop)
                    #row.start = 0
                elif row.start < 0 or row.start >= M:
                    raise IndexError('Index out of bounds')
                if row.stop == None:
                    row = slice(row.start,M)
                    #row.stop = M
                elif row.stop < 0 or row.stop >= M:
                    raise IndexError('Index out of bounds')
                if  row.start >= row.stop:
                    raise IndexError('')
        elif hasattr(row, '__len__'):
            row = np.unique(np.sort(row))
            if row[0] < 0 or row[0] >= M:
                raise IndexError('Index out of bounds')
        
        if isinstance(col,INT_TYPES):
            if col < 0 or col >=  N:
                raise IndexError('Index out of bounds')
        elif isinstance(col,slice):
            if not col==slice(None):
                if col.start == None:
                    col = slice(0,col.stop)
                    #col.start = 0
                elif col.start < 0 or col.start >= N:
                    raise IndexError('Index out of bounds')
                if col.stop == None:
                    col = slice(col.start,N)
                elif col.stop < 0 or col.stop >= N:
                    raise IndexError('Index out of bounds')
                if  col.start >= col.stop:
                    raise IndexError('')
        elif hasattr(col, '__len__'):
            col = np.unique(np.sort(col))
            if col[0] < 0 or col[0] >= N:
                raise IndexError('Index out of bounds')
                
        return row,col            
