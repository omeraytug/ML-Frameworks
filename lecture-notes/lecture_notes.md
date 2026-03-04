# L1

- scikitlearn: for more traditional usage. (Example: förklarbara matematiska modeller). LinReg, KNN, LogReg
- tensorflow: huvudval vid DL. Mer för produktion. Eager vs graph execution
- pytorch: huvud vid DL. Mer för forskning. Eager vs graph executiion

## Eager vs Graph Execution

- Eager: Kod körs rad för rad. (Kod som vi är vana vid)
- Graph: snabbare, mer effektivt. för produktion

## Supervised vs Unsupervised Learning (Typically within Traditional ML)

- Supervised: we have a label, we have told the model what the result is. Often classification. We send the X and Y.
- Unsupervised: we have no label. "matar in rådata". We let the the model find the relation itself. often a part of the EDA. Clustering. We send the X

## Three Domains - DL and RL could be seen as subdomains

- Machine Learning: allt dator lärande
- Deep Learning: train from exemple. mystistkt/black box. kasta in massor av data. efter ett tag blir den duktig. you train the model many times. we send the X and Y (often). Neural network.
- Reinforcement Learning: träna mot sig själv. belönas för bra. straffas med dåligt. (Exempel: chess, köra bil - om du vinner schack spelet eller förlorar så lär det sig: om du vinner så är det bra om du förlorar år det dåligt. self driving cars)

# L2

- Cosine similarity: hur lika två vektorer är.
- I regel kör PyTorch kod i eager stil (en rad i taget). Om man använder torch.compile, så kör den i graph-stil.

# L3

- Lecture 3 content: All 3 labs (more focus on the Lab 1). Some theory. Meta learning(?).
- Deadlines: Lab 1: Feb 13th - Lab 2: Feb 27th - Lab 3: March 12nd.

# L4

- F1 score a performance metric for classification models, such as LogReg and SVM.

# L5

## Evaluation Metrics

### Classification: Divides data into classes. Each prediction is either correct or incorrect. (from the model).

|                     | Actual: Positive                                            | Actual: Negative                                                |
| ------------------- | ----------------------------------------------------------- | --------------------------------------------------------------- |
| Predicted: Positive | True Positive (TP) (have diabates, we predicted have)       | False Positive (FP) (dont have diabates, we predicted have)     |
| Predicted: Negative | False Negative (FN) (have diabates, we predicted dont have) | True Negative (TN) (dont have diabates, we predicted dont have) |

- Accuracy: the proportion of total predictions that are correct. Accuracy can be misleading for imbalanced datasets, as it may appear high (e.g. 95%) even when the model misses positive cases (false negatives).

$$Accuracy = \frac{TP + TN}{TP + FP + FN + TN}$$

- Precision: Precision measures how many of the instances predicted as positive are actually positive.

$$Presicion: \frac{TP}{TP + FP}$$

- Recall: Measures how many of the actual positive cases are correctly identified by the model.

$$Recall: \frac{TP}{TP + FN}$$

- F1-Score: F1-score is a metric that combines precision and recall into a single value by taking their harmonic mean.

$$F1 = \frac{2PR}{P + R}$$

---

- Standardization: Standardization scales the data to have a mean of 0 and a standard deviation of 1.

- K-Means: K-Means clusters data into K groups by minimizing the distance between data points and their cluster centroids.

- Inertia: Inertia is the sum of the squared distances of each data point to its closest centroid.

- PCA: PCA reduces the dimensionality of the data by projecting it onto a lower-dimensional space while preserving as much of the original variance as possible.

- Elbow Method: The elbow method is a technique to find the optimal number of clusters. It plots the inertia (the sum of the squared distances of each data point to its closest centroid) against the number of clusters. The optimal number of clusters is the one where the inertia starts to decrease slowly.

# L6

### Neural Network Structure

- A neural network is a machine learning model composed of layers of connected computational units called nodes (or neurons).
- The basic structure consists of: 1. Input Layer / 2. Hidden Layer(s) / 3. Output Layer.
- Information flows from the input layer → through hidden layers → to the output layer.

---

### Input Layer

- The input layer receives the features (variables) from the dataset and passes them to the next layer. Each node represents one feature. Usually no computation, only forwards the data. Number of nodes = number of features.

---

### Nodes (Neurons)

- A node (neuron) is the basic computational unit of a neural network. Each neuron computes a weighted sum of the inputs and applies an activation function.

### Neuron Computation

A neuron computes a weighted sum of its inputs and applies an activation function.

#### Computation

$$
z = w_1x_1 + w_2x_2 + ... + w_nx_n + b
$$

$$
a = f(z)
$$

#### Where

- `x` = input values
- `w` = weights
- `b` = bias
- `f` = activation function

#### Common Activation Functions

- ReLU
- Sigmoid
- Tanh
- Softmax

---

### Hidden Layers

Hidden layers are located between input and output layers. They recieve inputs from the previous layer, compute weighted sums, apply activation functions, pass results to the next layer. These layers learn patters and relationships in the data.

Hidden layers are what make the model "deep".

They allow the network to learn:

- **Non-linear transformations**
- **Feature representations (representation learning)**

Hidden layers transform the input data into more useful representations for the final prediction.

Neural networks are not a complete "black box" mathematically, but interpreting what each hidden layer learns can be difficult.

## The **number of nodes in hidden layers is a hyperparameter**, which we choose when designing the model.

### Output Layer

- The output layer produces the final prediction, number of output nodes depends on the task.

#

Example:

Dataset contains animal traits (weight, height, color) and 5 possible animal species.

Input nodes = 3 (one for each feature)  
Output nodes = 5 (one for each species)

Network structure:

Input (3) → Hidden layer(s) → Output (5)

#

### Weights and Bias

Every connection between two nodes in a neural network has a **weight**.

If we have:

- 2 input nodes
- 5 nodes in the next layer (e.g., a hidden layer)

Then the number of weights is:

2 × 5 = **10 weights**

This is because **each node connects to every node in the next layer**.

---

### Neuron Computation

A neuron computes a weighted sum of its inputs and then applies an activation function.

z = w1x1 + w2x2 + ... + wnxn + b  
a = f(z)

Where:

- **x** = input values
- **w** = weights
- **b** = bias
- **f** = activation function

---

### Weight (w)

A **weight** determines how important an input is for the neuron.

- Large weight → strong influence on the output
- Small weight → weak influence
- Negative weight → inverse influence

Weights are **learned during training**.

---

### Bias (b)

The **bias** is an additional parameter added to the weighted sum.

z = Wx + b

Bias allows the neuron to **shift the activation function**, which helps the model better fit the data.

Bias is also **learned during training**.

---

### Forward Pass

Forward pass:

1. Take the input
2. Multiply inputs by weights and sum them
3. Add the bias
4. Apply the activation function
5. Pass the result to the next layer
6. Repeat through all layers until the output (prediction) is produced

---

### Backpropagation

In supervised learning we have the **true labels (targets)**.

Backpropagation steps:

1. Calculate the **loss** (difference between prediction and true value)
2. Compute the **gradients** of the loss with respect to the weights
3. **Update the weights**

This is done using **gradient descent**.

Backpropagation is an efficient method for computing gradients **by propagating the error backward through the network**.

---

### Tensors

A **tensor** is a multi-dimensional array used to store data in machine learning and deep learning.

| Dimension | Name   |
| --------- | ------ |
| 0D        | Scalar |
| 1D        | Vector |
| 2D        | Matrix |
| 3D+       | Tensor |

### Examples

0D: 7

1D: [5, 2]

2D:
[[2, 3],
 [5, 7]]

3D:
[
[[1,4],[1,2]],
[[8,3],[1,2]]
]

Higher dimensions follow the same structure with additional nested arrays.

---

### Normalization

#### Why normalize data?

- More stable training
- Lower risk of exploding or vanishing gradients
- Faster convergence during training

#### Standardization

One common normalization method is **standardization**:

X_scaled = (x − μ) / σ

Where:

- x = original value
- μ = mean of the dataset
- σ = standard deviation

This transforms the data so that it has:

- Mean ≈ 0
- Standard deviation ≈ 1

#### In PyTorch:

Normalization can be applied using:

transforms.Normalize(mean, std)

The **mean and standard deviation should match the dataset**.

---

### Overfitting and Regularization

If a model is trained for too long:

- It fits the training data too closely
- It generalizes poorly to new data

This is called **overfitting**.

Common Solutions

- **L2 regularization (weight decay)**
- **Dropout**
- **Early stopping**

### Regularization

Regularization adds a **penalty term** to the loss function:

Loss_total = Loss_data + λ||W||²

Where:

- Loss_data = loss from the predictions
- W = model weights
- λ = regularization strength

This encourages the model to keep **smaller weights**, which helps reduce overfitting.

# L11

- Transfer learning: Transfer learning is the process of reusing a model trained on one task (source task) for another related task (target task), typically by fine-tuning the model or reusing some of its layers instead of training a new model from scratch.

- In transfer learning, we freeze the weights of some layers of a pretrained model so that their learned parameters are not updated during training on the new task. This allows the model to retain the general features it learned from the source task, while only the remaining (unfrozen) layers are fine-tuned to adapt to the target task - we typically freeze the early (lower) layers.
