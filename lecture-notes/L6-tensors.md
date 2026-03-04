# Tensors, Reshape, Broadcasting

## Tensors

A **tensor** is a multi-dimensional data structure.

A:
- Scalar = 0D tensor
- Vector = 1D tensor
- Matrix = 2D tensor
- Higher dimensions = 3D, 4D, nD tensors

You can think of them as generalized matrices.
In PyTorch, tensors are used because they:
- Can run efficiently on CPU and GPU
- Support automatic differentiation (autograd)
- Are the fundamental type for all neural networks

## Shape and Creating Tensors
Example:
```python
x = torch.randn(2,3)
```
Shape = `(2,3)`
This means:
- 2 rows
- 3 columns
- total 6 elements
Number of elements = product of dimensions

## Reshape

You can only reshape if the number of elements remains unchanged.
Example:
```python
x = torch.rand(2,3) # 6 elements
y = x.reshape(3,2) # also 6 elements
```
This works because:
2 x 3 = 3 x 2 = 6

Reshape does **not** create new values. It's just a reorganization of the same data in memory.

Example:
```python
(2,3)
[[1,2,3],
[4,5,6]]
```
Can become:
```python
(3,2)
[[1,2],
[3,4],
[5,6]]
```
But:
```python
x.reshape(4,2)
```
Gives an error because 4 x 2 = 8 ≠ 6

## Special Case -1

You can let PyTorch calculate one dimension:
```python
x.reshape(-1, 2)
```
This means:
- I want 2 columns
- Calculate how many rows are needed

If x has 6 elements, the result is `(3, 2)`

**Important**: Only **one** dimension can be -1

## Multi-dimensional Tensors
Example:
```python
torch.randn(3,4,10)
```
Means:
- 3 blocks
- each block has 4 rows
- each row has 10 numbers

Total number of elements:
3 x 4 x 10 = 120

Tensors work mathematically the same regardless of number of dimensions.

## Why Reshape is Needed in Neural Networks
A `nn.Linear` expects input in the form: `(batch_size, features)`, i.e., 2D. But image data is often `(batch_size, channels, height, width)`, i.e., 4D. Then you need to "flatten" all dimensions except batch.

Example:
```python
(3,4,10) -> (3,40)
```
We keep only the number of data points (3), but convert 4x10 -> 40 features. This is called **flattening**.
