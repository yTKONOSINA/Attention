import math

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
    
    def __mul__(self, other):
        """Scalar multiplication"""
        if isinstance(other, (int, float)):
            def multiply(data, scalar):
                if not isinstance(data, list):
                    return data * scalar
                return [multiply(x, scalar) for x in data]
            return Tensor(multiply(self.tensor, other))
        else:
            raise TypeError(f"Only scalar multiplication is supported")

    def __matmul__(self, other: "Tensor") -> "Tensor":
        """
            This function does not implemnt broadcasting!
            The dimensions before the last two must be identical!
            Args:
                A, B : Tensor
            Returns:
                A @ B with the shape: (..., n, m)
                (..., n, p) @ (..., p, m) -> (..., n, m)
                (..., n, p) @ (p, m) -> (..., n, m) batch matrix multiplication
        """
        if len(other.shape) == 2 and len(self.shape) > 2:
            assert self.shape[-1] == other.shape[-2], \
            f"Last dim of first must match second-to-last of second"
            return Tensor(self._matmul(self.tensor, other.tensor))

        assert self.shape[:-2] == other.shape[:-2], \
        "Batch dimensions do not match!"
        assert self.shape[-1] == other.shape[-2], \
        "Cannot multiply the two tensors, check the final dimensions"
        return Tensor(self._matmul(self.tensor, other.tensor))

    def _matmul(self, a, b):
        a_shape = self._get_shape(a)
        b_shape = self._get_shape(b)
        
        if len(a_shape) == 2 and len(b_shape) == 2:
            m, n = len(a), len(a[0])
            n, p  = len(b), len(b[0])
            return [
                [
                    sum(a[i][k] * b[k][j] for k in range(n))
                    for j in range(p)
                ]
                for i in range(m)
            ]
        elif len(a_shape) > 2 and len(b_shape) == 2:
            return [self._matmul(x, b) for x in a]
        elif len(a_shape) == len(b_shape) and len(a_shape) > 2:
            return [self._matmul(x, y) for x, y in zip(a, b)]
        else:
            raise ValueError(f"Cannot multiply tensors with shapes")


    def cat(self, other : "Tensor") -> "Tensor":
        pass

    def transpose_2d(self) -> "Tensor":
        assert len(self.shape) == 2, "Only transpose 2d tensors"
        return Tensor([list(row) for row in zip(*self.tensor)])
    
    def reshape(self, new_shape : tuple) -> "Tensor":
        """
            The product of the current shape should be equal to the
            product of the new shape. 
            First the tensor is flattened and then it is reshaped
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

    def permute(self, dims : tuple) -> "Tensor":
        """
            Reorder the dimensions of the tensor
            Example:
                x.shape = (1, 2, 3)
                x.permute((2, 1, 0)) -> (3, 2, 1)
            Supports tensors only up to 4 dimensions
        """
        assert len(dims) == len(self.shape), \
        "The number of dimensions must match"

        assert sorted(dims) == list(range(len(dims))), \
        "dims must be a permuation of tensor axes"

        if len(dims) == 2:
            if dims == (1, 0):
                return self.transpose_2d()
            return self
    
        if len(dims) == 3:
            new_shape = tuple(self.shape[i] for i in dims)
            new_tensor = [[[0 for _ in range(new_shape[2])]
                           for _ in range(new_shape[1])]
                           for _ in range(new_shape[0])]

            for i in range(new_shape[0]):
                for j in range(new_shape[1]):
                    for k in range(new_shape[2]):
                        old_idx = [i, j, k]
                        old_i = old_idx[dims.index(0)]
                        old_j = old_idx[dims.index(1)]
                        old_k = old_idx[dims.index(2)]
                        new_tensor[i][j][k] = self.tensor[old_i][old_j][old_k]
            return Tensor(new_tensor)

        if len(dims) == 4:
            new_shape = tuple(self.shape[i] for i in dims)
            new_tensor = [[[[0 for _ in range(new_shape[3])]
                            for _ in range(new_shape[2])]
                            for _ in range(new_shape[1])]
                            for _ in range(new_shape[0])]

            for i in range(new_shape[0]):
                for j in range(new_shape[1]):
                    for k in range(new_shape[2]):
                        for l in range(new_shape[3]):
                            old_idx = [i, j, k, l]
                            old_i = old_idx[dims.index(0)]
                            old_j = old_idx[dims.index(1)]
                            old_k = old_idx[dims.index(2)]
                            old_l = old_idx[dims.index(3)]
                            new_tensor[i][j][k][l] = self.tensor[old_i][old_j][old_k][old_l]
            return Tensor(new_tensor)

        raise NotImplementedError("permute supports up to 4D tensors only.")

    # bmm function

    def transpose(self, dim1 : int, dim2 : int) -> "Tensor":
        dims = list(range(len(self.shape)))
        dims[dim1], dims[dim2] = dims[dim2], dims[dim1]
        return self.permute(tuple(dims))
    
    def softmax(self, dim : int = -1) -> "Tensor":
        """
            Applies softmax along a given dimension.
            Works for tensors up to the 4th dimension.
        """

        eps = 1e-12

        def softmax_1d(array):
            if all(x == float("-inf") for x in array):
                return [0.0 for _ in array]
            mx = max(array)
            exps = [math.exp(x - mx) for x in array]
            s = sum(exps)
            return [e / (s + eps) for e in exps]

        def apply_softmax(data, curr_dim, tar_dim):
            if not isinstance(data, list):
                return data
            if tar_dim == curr_dim:
                if isinstance(data[0], list):
                    return [softmax_1d(sample) if not isinstance(sample, list) else apply_softmax(sample, curr_dim + 1, tar_dim) for sample in data]
                else:
                    return softmax_1d(data)
            else:
                return [apply_softmax(sample, curr_dim + 1, tar_dim) for sample in data]
        
        if dim < 0:
            dim += len(self.shape)
        
        if len(self.shape) == 2 and dim == 1:
            return Tensor([softmax_1d(row) for row in self.tensor])
        else:
            return Tensor(apply_softmax(self.tensor, 0, dim))

    def masked_fill(self, mask: "Tensor", value: float) -> "Tensor":
        """
            Replaces elements where mask != 0 with 'value'
        """
        def fill(data, mask_data):
            if not isinstance(data, list):
                return value if mask_data else data
            return [fill(d, m) for d, m in zip(data, mask_data)]

        return Tensor(fill(self.tensor, mask.tensor))

if __name__ == "__main__":
    # a = Tensor([[1, 2], [3, 4]])
    # print(a.shape)
    # b = Tensor([[5, 6], [7, 8]])
    # print((a + b).tensor)
    # print((a @ b).tensor)
    # print(a.flatten().shape)
    # a = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
    # print(a.reshape((1, 1, 1, 2, 2, 2)).tensor)

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    #print(x.transpose_2d().transpose_2d().tensor)
    print(x.softmax_dim_0().tensor)