
# Semantic Equivalence 

I became interested in indentifying novel written ideas amoung a many voices. A simple example might be: "The sky is blue" is equivalent to "The color of the sky is blue". Aggregatting ideas reduces distractions related to how an individual idea was poosed. Moreover, aggregation adds insights into the popularity of specific ideas. To tell if two ideas are equivalent is a lofty goal. I starterd with simple one or two sentences from quora. The data consists of pairs of questions generously made availbe by [data.quora.com](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)

### Applications

The use case is to streamline customer support by aggregating customer request for help. With the model in production a  business can better assist customers by quickly matching the posed question to a list of pre-answered questions. 

Another use case would be for message board type sites. The model can check new posts to exsiting post and determine if equivalent post has been made. This aggregates response in once place for a better user expirence.

### Data Understanding

The dataset consists of over 400,000 lines of potential question duplicate pairs. Each line contains IDs for each question in the pair, the full text for each question, and a binary value that indicates whether the line truly contains a duplicate pair. Here are a few sample lines of the dataset:

| Question1 | Question2 | Equivalent?
|----------|----------|----------|
|What are natural numbers?  | What is a least natural number? | No
|Which Pizzas are the most popularity ordered pizzas on Domino's menu? | How many calories does a Dominos pizza have? | No
|How do you start a bakery? | How can one start a bakery business? | Yes
|Should I learn python or Java First? | If I had to choose between learning Java and Python, what should I choose to learn first? | Yes


Here are a few important things to keep in mind about this dataset:

- The dataset set is imbalanced with many more true examples of duplicate pairs than non-duplicates.

- The distribution of questions in the dataset should not be taken to be representative of the distribution of questions asked on Quora. This is, in part, because of the combination of sampling procedures and also due to some sanitization measures that have been applied to the final dataset (e.g., removal of questions with extremely long question details).

- The ground-truth labels contain some amount of noise: they are not guaranteed to be perfect.

### Data Preperation

The data was provided in a clean format. It was taken from a csv into pandas dataframe. The 3 / 400,0000 pairs had null values and were dropped


### Neural Networks

#### Word Encoding

The model with the best preformance used pre-trained word vectors. This data is made available under the Public Domain Dedication and License v1.0 whose full text can be found at: http://www.opendatacommons.org/licenses/pddl/1.0/. Specifically the 300 dimensianl space vectors.

#### Conjoined Networks

![Typical Conjoined Network --placeholder](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

The most successful models were based on what I call joincoined neural networks (the literature calls siamese neural networks). These networks have identical encoding layers and indentical neural netwworks. The seperate paths are conjoined and passed through a layer with distance metric calculations. The tensors are then passed through a dense and drop out layers before output.  

Conjoine model2 keras deployed gated recurrent unit.

### Cosine Similarity Modeling

Modeling options include options for vectorization and distance calculations. Vectorization are built ontop of Sklearns TFIDF Vectorization and Count Vectorization. Distance options are cosine simillarity, jaccard distance and euclidian distance.


### Evaluation

The current best model is the conjoined_model2 preforming at a 78% accuracy with Area under ROC of 90%

This model can be accessed [from my googledrive](https://drive.google.com/file/d/1DYECLvdwC123LthIj0lHL-KjddCuEnKG/view?usp=sharing)


### Instructions
>>>>>>> 7b191deb115be1d440003ff160fa8a49bda6dea8

