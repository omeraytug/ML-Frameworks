# Simple Neural Network

## Linear Layer - What It Does
`nn.Linear(4,10)` creates:
- Weight matrix with shape (10,4)
- Bias with shape (10,)

It performs the computation: $y = xW^T + b$

For each data point:
- 4 numbers in
- 10 new numbers out

Each output is a weighted sum of all input values + bias.

## ReLU
`nn.ReLU()` Function: `ReLU(x) = max(0, x)`

It makes the model non-linear. Without non-linearity, multiple Linear layers in sequence would be mathematically equivalent to a single Linear layer.

## Final Layer and Logits
`nn.Linear(10,3)`
- 10 numbers in
- 3 numbers out

These three numbers are called **logits** (output)

## CrossEntropyLoss
`criterion = nn.CrossEntropyLoss()`

It:
1. Applies `Softmax` internally
2. Computes negative log-likelihood
3. Compares with the correct class (integer index)

Important:
- Targets must be dtype=torch.long
- You should **not** apply Softmax yourself before the loss

## Data Types
```python
x = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
```
Why?
- The network uses float
- CrossEntropyLoss requires integer classes (long)

## Optimization
`optimizer = optim.Adam(model.parameters(), lr=0.01)`

- `model.parameters()` = all weights and biases
- `lr` = step size (learning rate)
- **Adam** = adaptive gradient-based method

## Training Loop - What Happens

```python
for _ in range(n_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
```

### 1. optimizer.zero_grad()
Each parameter in the model has:
- `param.data` = the weight itself
- `param.grad` = the gradient from the previous backprop

PyTorch **adds gradients** each time you run `.backward()`, so without `zero_grad()` the gradients would accumulate.

Mathematically: $g_{total} = g_1 + g_2 + g_3 + ...$

Sometimes this is intentional (e.g., gradient accumulation), but not in standard training.

### 2. Forward Pass
```python
outputs = model(X)
```
Here happens:
1. Data passes through first linear: $z_1 = XW_1^T + b_1$
2. ReLU: $a_1 = \max(0, z_1)$
3. Final linear: $\text{logits} = a_1W_2^T + b_2$

PyTorch simultaneously builds a **computational graph**.

This means each operation saves:
- Which tensors are used
- How they are used
- How the gradient should be computed backwards

This is the foundation of **automatic differentiation**.

### 3. Loss Calculation
```python
loss = criterion(outputs, y)
```
For classification with CrossEntropyLoss:
Internally: $Softmax(\text{logits})$ and then: $-\log(p_{\text{correct}})$ for **each** data point.

The loss is a single scalar value: `loss.shape == torch.Size([])`. This is important: Backprop always starts from a scalar.

### 4. Backpropagation
```python
loss.backward()
```
This is the central mechanism.

PyTorch computes: $\frac{\partial L}{\partial w}$ for each parameter $w$.

**How?**

Chain rule: $\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \text{logits}} \cdot \frac{\partial \text{logits}}{\partial a_1} \cdot \frac{\partial a_1}{\partial z_1} \cdot \frac{\partial z_1}{\partial W_1}$

This happens backwards through the graph. The result is stored in `param.grad`.

No weights are updated here, only gradients are computed.

### 5. optimizer.step()
Now the weights are updated.

A simplified gradient descent: $w := w - \text{lr} \cdot \frac{\partial L}{\partial w}$

Adam does more:
- maintains rolling average of gradients
- maintains rolling average of squared gradients
- normalizes the steps

But the principle is the same - we move in the direction that decreases loss.

## What Happens Over Multiple Epochs?

An epoch means: The entire dataset passes through the model once.

After each epoch:
- Weights are slightly better adapted.
- Loss should decrease (if learning rate is reasonable)

## What Does the Gradient Mean (Intuitively)?
If: $\frac{\partial L}{\partial w} = 5$
This means:
- If we increase the weight by 1, the loss increases by approximately 5.
- Therefore we want to decrease the weight.

If: $\frac{\partial L}{\partial w} = -2$
This means:
- If we increase the weight by 1, the loss decreases by 2.
- Therefore we want to increase the weight.

The gradient thus indicates:
- Direction
- Sensitivity

## Accuracy
```python
preds = torch.argmax(outputs, dim=1)
```
This selects: $\arg\max_j \text{logits}_j$
That is, the class with the highest raw score.

Accuracy: $\frac{\text{number of correct}}{\text{number of data points}}$
**BUT** - Accuracy says nothing about:
- how confident the model was
- class balance
- calibration

## System Understanding (Framework Level)

What makes PyTorch an ML framework:
1. Tensor computations
2. Automatic differentiation
3. Modular layer API (nn.Module)
4. Optimization algorithms
5. Computational graph

The training loop is essentially just:
- Build graph
- Compute loss
- Differentiate graph
- Update parameters
- Repeat

## Common Mistakes in Training Loops

1. Forgetting `zero_grad()`
2. Applying Softmax before CrossEntropyLoss
3. Wrong dtype on targets
4. Learning rate too high
5. Not using `model.train()` / `model.eval()` with dropout/batchnorm

## Training Loop Summary

Each epoch does:
1. Approximates the gradients of the loss surface
2. Takes a small step in the negative gradient direction
3. Repeats until we reach:
   - local minimum
   - saddle point
   - or get stuck
