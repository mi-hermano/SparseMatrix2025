# Contains the SparseMatrix object
import numpy as np
tol = 1e-8  # Tolerance for zero values in the matrix


class SparseMatrix:
    def __init__(self, arr: np.ndarray = None):  # Sparsify matrix
        """
        Generates a Sparse Matrix using the Compressed Sparse Row (CSR) format
        with the following properties: \n
        V: Values of nonzero elements in matrix. \n
        col_index: Column indices for each nonzero element in matrix. \n
        row_counter: Number of nonzero elements in a row such that row i contains
        row_counter[i+1] - row_counter[i] elements (0-indexed). \n
        number_of_nonzero: Total number of nonzero elements in matrix. \n
        intern_represent: Sparse matrix compression format.

        :param arr: Numpy sparse matrix to be compressed
        """
        temp_V = []
        temp_col_index = []
        temp_row_counter = [0]
        temp_number_of_nonzero = 0  # can be further simplified later

        if arr is None:  # Create empty object
            self._V = np.array(temp_V)
            self._col_index = np.array(temp_col_index)
            self._row_counter = np.array(temp_row_counter)
            self._number_of_nonzero = temp_number_of_nonzero
            self._intern_represent = 'CSR'
            self._shape = np.array([0, 0])
            return

        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if abs(arr[i, j]) > tol:
                    temp_number_of_nonzero += 1
                    temp_V.append(arr[i, j])
                    temp_col_index.append(j)

            temp_row_counter.append(temp_number_of_nonzero)

        self._V = np.array(temp_V)
        self._col_index = np.array(temp_col_index)
        self._row_counter = np.array(temp_row_counter)
        self._number_of_nonzero = temp_number_of_nonzero
        self._intern_represent = 'CSR'
        self._shape = arr.shape # jag lade till denna /RS

    def __repr__(self):
        return(
            f"""
            NNZ: {self._number_of_nonzero}
            Values: {self._V}
            Columns: {self._col_index}
            Row Counter: {self._row_counter}
            """
               )

    @property
    def number_of_nonzero(self):
        return self._number_of_nonzero

    @property
    def V(self):
        return self._V

    @property
    def col_index(self):
        return self._col_index

    @property
    def row_counter(self):
        return self._row_counter

    @property
    def intern_represent(self):
        return self._intern_represent

    def edit(self, x, i, j):
        isOccupied = False
        nonZero = False
        V = self._V
        Col = self._col_index
        Row = self._row_counter
        
        if i > self._shape[0] - 1:
            self._col_index = np.append(self._col_index, j)
            self._V = np.append(self._V, x)
            print(np.size(self._row_counter))
            while np.size(self._row_counter) - 1 <= i:
                self._row_counter = np.append(self._row_counter, self._row_counter[-1])
            self._row_counter[-1] += 1
            
        else:
            row_start  = self._row_counter[i]
            row_end  = self._row_counter[i+1]
            
            if abs(x) > tol:
                nonZero = True

            if j in self._col_index[row_start:row_end]:
                isOccupied  = True
            
            if isOccupied and nonZero:
                workCol = Col[row_start:row_end]
                workV = V[row_start:row_end]
                n = 0
                while j > workCol[n]:
                    n += 1
                workV[n] = x
                self._V = np.concatenate((V[0:row_start], workV, V[row_end:len(V)]))
                
            elif not isOccupied and nonZero:
                workCol = Col[row_start:row_end]
                workV = V[row_start:row_end]
                
                if (workCol.size == 0) or (j > np.max(workCol)):
                    workCol = np.append(workCol, [j])
                    workV = np.append(workV, [x])
                else:
                    n = 0
                    while j > workCol[n]:
                        n += 1
                    workCol = np.insert(workCol, n, j)
                    workV = np.insert(workV, n, x)
                
                for a in range(i + 1, len(Row)):
                    Row[a] += 1
                    
                self._col_index = np.concatenate((Col[0:row_start], workCol, Col[row_end:len(Col)]))
                self._V = np.concatenate((V[0:row_start], workV, V[row_end:len(V)]))
                self._row_counter = Row
                self._number_of_nonzero += 1
            
            elif isOccupied and not nonZero:
                workCol = Col[row_start:row_end]
                workV = V[row_start:row_end]
                n = 0
                while j > workCol[n]:
                    n += 1
                workCol = np.delete(workCol, n)
                workV = np.delete(workV, n)
                for a in range(i + 1, len(Row)):
                    Row[a] -= 1
                self._col_index = np.concatenate((Col[0:row_start], workCol, Col[row_end:len(Col)]))
                self._V = np.concatenate((V[0:row_start], workV, V[row_end:len(V)]))
                self._row_counter = Row
                self._number_of_nonzero -= 1
                
            while self._row_counter[-1] == self._row_counter[-2]:
                self._row_counter = np.delete(self._row_counter, -1)
            
            
        self._shape = (np.size(self._row_counter) - 1, int(np.max(self._col_index)) + 1)
    
    def __add__(self, other):
        if self._intern_represent != other._intern_represent:
            raise ValueError("Cannot add matrices with different representations")
        if self._shape != other._shape:
            raise ValueError("Cannot add matrices with different dimensions")
        
        sum_V = []
        sum_col_index = []
        sum_row_counter = [0]
        
        for i in range(self._shape[0]):
            row_start_self = self._row_counter[i]
            row_end_self = self._row_counter[i + 1]
            row_start_other = other._row_counter[i]
            row_end_other = other._row_counter[i + 1]
            
            row_self = {self._col_index[i]: self._V[i] for i in range(row_start_self, row_end_self)}
            row_other = {other._col_index[i]: other._V[i] for i in range(row_start_other, row_end_other)}
            
            row_sum = {}
            for j in row_self:
                row_sum[j] = row_self[j]
            for j in row_other:
                if j in row_sum:
                    row_sum[j] += row_other[j]
                else:
                    row_sum[j] = row_other[j]
            for j in row_sum:
                if abs(row_sum[j]) > tol:
                    sum_V.append(row_sum[j])
                    sum_col_index.append(j)
            sum_row_counter.append(len(sum_V))
        
        sum = SparseMatrix(np.zeros((self._shape[0], self._shape[1])))
        sum._V = np.array(sum_V)
        sum._col_index = np.array(sum_col_index) 
        sum._row_counter = np.array(sum_row_counter)
        sum._number_of_nonzero = len(sum_V)
        sum._intern_represent = 'CSR'
        sum._shape = self._shape
        return sum

    @staticmethod
    def manual_toeplitz(n: int):
        if n < 0:
            raise ValueError("Number of rows must be a positive integer!")

        result = SparseMatrix()  # Create empty object to fill and return

        if n == 1:  # Trivial case
            result._V = np.array([2])
            result._col_index = np.array([0])
            result._row_counter = np.array([0, 1])
            result._number_of_nonzero = 1
            result._intern_represent = 'CSR'
            result._shape = np.array([1, 1])
            return result

        # Cases where n > 1 are generated procedurally

        temp_V = []
        temp_col_index = []
        temp_row_counter = [0]
        temp_number_of_nonzero = 0

        head = [2, -1]
        spine = [-1, 2, -1]
        tail = [-1, 2]

        columnpointer = 0

        # Create head of matrix
        temp_number_of_nonzero += 2
        temp_V += head
        temp_col_index += [columnpointer, columnpointer + 1]
        temp_row_counter.append(temp_number_of_nonzero)

        for i in range(0, n-2):  # Create spine of matrix
            temp_number_of_nonzero += 3
            temp_V += spine
            temp_col_index += [columnpointer, columnpointer+1, columnpointer+2]
            temp_row_counter.append(temp_number_of_nonzero)
            columnpointer += 1

        # Create tail of matrix
        temp_number_of_nonzero += 2
        temp_V += tail
        temp_col_index += [columnpointer, columnpointer+1]
        temp_row_counter.append(temp_number_of_nonzero)

        result._V = np.array(temp_V)
        result._col_index = np.array(temp_col_index)
        result._row_counter = np.array(temp_row_counter)
        result._number_of_nonzero = np.array(temp_number_of_nonzero)
        result._intern_represent = 'CSR'
        result._shape = np.array([n, n])
        return result

    @staticmethod
    def toeplitz(n: int):  # alternative solution
        if n < 0:
            raise ValueError("Number of rows must be a positive integer!")

        result = SparseMatrix()  # Create empty object to fill and return

        if n == 1:  # Trivial but strange case
            result._V = np.array([2])
            result._col_index = np.array([0])
            result._row_counter = np.array([0, 1])
            result._number_of_nonzero = 1
            result._intern_represent = 'CSR'
            result._shape = np.array([1, 1])
            return result

        # Freaky generation method
        temp_V = [2] + [-1, -1, 2] * (n-1)
        temp_col_index = [0] + [x for xs in ([i+1, i, i+1] for i in range(n-1)) for x in xs]
        temp_row_counter = [0] + [2+3*i for i in range(n-1)] + [4+3*(n-2)]
        temp_number_of_nonzero = temp_row_counter[-1]

        result._V = np.array(temp_V)
        result._col_index = np.array(temp_col_index)
        result._row_counter = np.array(temp_row_counter)
        result._number_of_nonzero = np.array(temp_number_of_nonzero)
        result._intern_represent = 'CSR'
        result._shape = np.array([n, n])
        return result

