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
        return Tensor()

    # overload @
    # concatination of two tensors
    # transposition
    # reshaping
    # bmm function

if __name__ == "__main__":
    a = Tensor([1, 2, 3])
    print(a.shape)
    b = Tensor([4, 5, 6])
    print(a + b)