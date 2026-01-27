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