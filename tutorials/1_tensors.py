"""PyTorch tensors tutorial."""

import torch
import numpy as np


def main():
    # Initialize tensors directly from data
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)

    # Initialize from numpy array
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)

    # Initialize from another tensor
    x_ones = torch.ones_like(x_data)  # Should be the same size as x_data
    print(f'Ones Tensor: \n{x_ones}\n')
    x_rand = torch.rand_like(x_data, dtype=torch.float)
    print(f'Random Tensor: \n{x_rand}\n')
    
    # Note that the 'like' functions also retain data type properties

    # Other initialization options:
    shape = (2, 3)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f'Random Tensor: \n{rand_tensor}\n')
    print(f'Ones Tensor: \n{ones_tensor}\n')
    print(f'Zeros Tensor: \n{zeros_tensor}\n')
    
    # Tensor attributes:
    tensor = torch.rand(3, 4)
    print(f'Tensor shape: {tensor.shape}')
    print(f'Tensor datatype: {tensor.dtype}')
    print(f'Tensor device: {tensor.device}')
    
    # Operations on tensors

    # Indexing and slicing
    tensor = torch.ones(4, 5)
    print('First row: ', tensor[0])
    print('First column: ', tensor[:, 0])
    tensor[:, -1] = 10
    print('Last column: ', tensor[:, -1])
    tensor[:, 1] = 0
    print(tensor)
    
    # Joining tensors
    t1 = torch.cat([tensor, tensor, tensor], dim=1)
    print(t1)
    # Note that 'cat' does not create a new dimension (as opposed to 'stack')

    # Arithmetic operations
    y1 = tensor @ tensor.T  # (4, 5) x (5, 4)
    y2 = tensor.matmul(tensor.T)  # Same thing
    
    y3 = torch.rand_like(tensor[:, 0:4])  # Preallocate memory
    torch.matmul(tensor, tensor.T, out=y3)  # Assign output to variable

    # y1, y2, and y3 should all be the same
    print(f'y1: \n{y1}\n')
    print(f'y2: \n{y2}\n')
    print(f'y3: \n{y3}\n')

    # Elementwise products
    z_1 = tensor * tensor
    z_2 = tensor.mul(tensor)

    z_3 = torch.rand_like(tensor)
    torch.mul(tensor, tensor, out=z_3)

    # z_1, z_2, and z_3 should all be the same
    print(f'z_1: \n{z_1}\n')
    print(f'z_2: \n{z_2}\n')
    print(f'z_3: \n{z_3}\n')

    # Single-element tensors
    agg = tensor.sum()
    agg_item = agg.item()
    print(agg, type(agg))
    print(agg_item, type(agg_item))

    # In-place operations
    print(tensor, '\n')
    tensor.add_(5)
    print(tensor)

    # Bridge with NumPy
    # Tensors on the CPU and NumPy arrays can share the same memory locations
    t = torch.ones(5)
    print(f't: {t}')
    n = t.numpy()
    print(f'n: {n}')

    t.add_(1)  # In-place add
    print(f't: {t}')
    print(f'n: {n}')

    n = np.ones(5)
    t = torch.from_numpy(n)
    np.add(n, 1, out=n)  # In-place numpy operation
    print(f't: {t}')
    print(f'n: {n}')

    return None


if __name__ == '__main__':
    main()

