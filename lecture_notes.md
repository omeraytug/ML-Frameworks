# L1 #
- scikitlearn: for more traditional usage. (Example: förklarbara matematiska modeller). LinReg, KNN, LogReg 
- tensorflow: huvudval vid DL. Mer för produktion. Eager vs graph execution
- pytorch: huvud vid DL. Mer för forskning. Eager vs graph executiion


## Eager vs Graph Execution ## 
- Eager: Kod körs rad för rad. (Kod som vi är vana vid)
- Graph: snabbare, mer effektivt. för produktion

## Supervised vs Unsupervised Learning (Typically within Traditional ML) ##
- Supervised: we have a label, we have told the model what the result is. Often classification. We send the X and Y.
- Unsupervised: we have no label. "matar in rådata". We let the the model find the relation itself. often a part of the EDA. Clustering. We send the X


## Three Domains - DL and RL could be seen as subdomains ##
- Machine Learning: allt dator lärande  
- Deep Learning: train from exemple. mystistkt/black box. kasta in massor av data. efter ett tag blir den duktig. you train the model many times. we send the X and Y (often). Neural network.
- Reinforcement Learning: träna mot sig själv. belönas för bra. straffas med dåligt. (Exempel: chess, köra bil - om du vinner schack spelet eller förlorar så lär det sig: om du vinner så är det bra om du förlorar år det dåligt. self driving cars)


# L2 #
- Cosine similarity: hur lika två vektorer är.
- I regel kör PyTorch kod i eager stil (en rad i taget). Om man använder torch.compile, så kör den i graph-stil.

# L3 # 
- Lecture 3 content: All 3 labs (more focus on the Lab 1). Some theory. Meta learning(?). 
- Deadlines: Lab 1: Feb 13th - Lab 2: Feb 27th - Lab 3: March 12nd. 


# L4 #
- F1 score a performance metric for classification models, such as LogReg and SVM. 
 
# L5 #
## Evaluation Metrics
### Classification: Divides data into classes. Each prediction is either correct or incorrect. (from the model). 

|                | Actual: Positive | Actual: Negative |
|----------------|------------------|------------------|
| Predicted: Positive | True Positive (TP) (have diabates, we predicted have)| False Positive (FP) (dont have diabates, we predicted have)|
| Predicted: Negative  | False Negative (FN) (have diabates, we predicted dont have)| True Negative (TN) (dont have diabates, we predicted dont have)|

- Accuracy: the proportion of total predictions that are correct. Accuracy can be misleading for imbalanced datasets, as it may appear high (e.g. 95%) even when the model misses positive cases (false negatives).

$$Accuracy = \frac{TP + TN}{TP + FP + FN + TN}$$

- Precision: Precision measures how many of the instances predicted as positive are actually positive.

$$Presicion: \frac{TP}{TP + FP}$$

- Recall: Measures how many of the actual positive cases are correctly identified by the model.

$$Recall: \frac{TP}{TP + FN}$$

- F1-Score: F1-score is a metric that combines precision and recall into a single value by taking their harmonic mean.

$$F1 = \frac{2PR}{P + R}$$

