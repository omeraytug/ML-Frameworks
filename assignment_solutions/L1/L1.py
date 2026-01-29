"""
Lektion 1 - ML-ramverk och arkitektur
Assignment: Frameworks, tensors, and execution models

Instructions:
1. Complete the tasks below with short, runnable code snippets
2. Run each section and observe the output
3. Comment your code to explain what each part does
4. Keep everything in this file unless stated otherwise
"""

# Task 1: Vector and matrix basics (NumPy)

import numpy as np
# TODO: Create two vectors (length 3) and compute:
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# - dot product
dot_product = np.dot(v1, v2) 
print("Dot product:", dot_product)

# - L2 norm
l2_v1 = np.linalg.norm(v1)
l2_v2 = np.linalg.norm(v2)

print("L2 norm v1:", l2_v1)
print("L2 norm v2:", l2_v2)

# - cosine similarity
cosine_similarity = dot_product / (l2_v1 * l2_v2)
print("Cosine similarity:", cosine_similarity)

# TODO: Create a 2x3 matrix and multiply it by a length-3 vector
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
matrix_vector_product = matrix @ v1
print("Matrix-vector product:\n", matrix_vector_product)

# Task 2: Eager vs graph execution
# TODO: Write a small function f(x) = x^3 + 2x
# TODO: Implement f(x) in ONE of:
# - PyTorch (eager)
# - TensorFlow with @tf.function (graph)
# - JAX with @jit (graph-like)
# TODO: Print the output and note how execution differs

# See L1.ipynb for the task solution



# Task 3: Framework comparison in code
# TODO: Using scikit-learn, load the iris dataset
from sklearn.datasets import load_iris
data = load_iris()

x = data["data"]
y = data["target"]

print(x)
print(y)


# TODO: Train a LogisticRegression model
# TODO: Train a tiny MLP (MLPClassifier) on the same data
# TODO: Compare accuracy and write 3-5 comments in code about:
# - speed
# - API ergonomics
# - when you would pick each approach

