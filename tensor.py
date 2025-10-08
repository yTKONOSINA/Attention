class Tensor:
    def __init__(self, data : list = None):
        self.tensor : list = []
        self.shape : tuple = ()
        if data:
            self.tensor = data
            self.shape = self._get_shape(data)
    
    def _get_shape(self, data):
        if isinstance(data, list):
            if len(data) == 0:
                return (0, )
            return (len(data), *self._get_shape(data[0]))
        else:
            return ()

    def __add__(self, other : "Tensor") -> "Tensor":
        assert (self.shape == other.shape), \
        "The tensors should be the same size"
        return Tensor(self._add_tensors(self.tensor, other.tensor))
    
    def _add_tensors(self, a, b):
        if not isinstance(a, list):
            return a + b
        return [self._add_tensors(x, y) for x, y in zip(a, b)]

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """
            This function does not implemnt broadcasting!
            The dimensions before the last two must be identical!
            Args:
                A, B : Tensor
            Returns:
                A @ B with the shape: (..., n, m)
                (..., n, p) @ (..., p, m) -> (..., n, m)
        """
        assert self.shape[:-2] == other.shape[:-2], \
        "Batch dimensions do not match!"
        assert self.shape[-1] == other.shape[-2], \
        "Cannot multiply the two tensors, check the final dimensions"
        return Tensor(self._matmul(self.tensor, other.tensor))

    def _matmul(self, a, b):
        if len(self._get_shape(a)) == 2 and len(self._get_shape(b)) == 2:
            m, n = len(a), len(a[0])
            n, p  = len(b), len(b[0])
            return [
                [
                    sum(a[i][k] * b[k][j] for k in range(n))
                    for j in range(p)
                ]
                for i in range(m)
            ]
        else:
            return [self._matmul(x, y) for x, y in zip(a, b)]


    def cat(self, other : "Tensor") -> "Tensor":
        pass

    def transpose(self) -> "Tensor":
        pass
    
    def reshape(self, new_shape : tuple) -> "Tensor":
        """
            The product of the current shape should be equal to the
            product of the new shape. 
            First the tensor is flattened and then it is reshape
        """
        total1 = 1
        for dim in self.shape:
            total1 *= dim
        
        total2 = 1
        for dim in new_shape:
            total2 *= dim
        
        assert total1 == total2, "Total number of elements must remain the same!"

        self.flatten()
        
        def _reshape(flat_list, shape):
            if len(shape) == 1:
                return flat_list[:shape[0]]

            step = len(flat_list) // shape[0]
            return [_reshape(flat_list[i*step:(i+1)*step], shape[1:])
                    for i in range(shape[0])]
            
        self.tensor = _reshape(self.tensor, new_shape)
        self.shape = new_shape

        return self

    def flatten(self) -> "Tensor":
        
        def _flatten_list(data):
            if not isinstance(data, list):
                return [data]
            flat = []
            for item in data:
                flat.extend(_flatten_list(item))
            return flat
        
        new_list = _flatten_list(self.tensor)
        self.tensor = new_list
        self.shape = (len(new_list),)
        return self

    def permute(self, new_shape : tuple) -> "Tensor":
        """
            Reorder the dimensions of the tensor
            Example:
                x.shape = (1, 2, 3)
                x.permute((2, 1, 0)) -> (3, 2, 1)
        """
        pass

    # bmm function

if __name__ == "__main__":
    a = Tensor([[1, 2], [3, 4]])
    print(a.shape)
    b = Tensor([[5, 6], [7, 8]])
    print((a + b).tensor)
    print((a @ b).tensor)
    print(a.flatten().shape)
    a = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    print(a.reshape((1, 1, 1, 2, 2, 2)).tensor)