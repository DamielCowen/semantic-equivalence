This project is a demonstration of advanced natural language processing (NLP) techniques using artificial neural nets in a deep learning model. 

There is an growing number of increasingly successful methods for assigning words to a location in vector space. The logical next step is to attempt this for sentences. This project is a humble step towards that lofty goal in determining if sentences are equivalent.

# Instructions

- clone the repo.
- download dependencies.
- run src/conjoined_nn_v2.py.

# Semantic Equivalence 

I became fascinated with NLP and specifically methods to determine the vector description of the word meanings.  Unsupervised deep learning techniques for learning global vector representations of words and found the solutions to be elegant.  These techniques create beautiful n-dimensional vector representation of words, organized in a natural and logical way (If not already familiar I highly recommend this [Stanford GloVe paper] (https://nlp.stanford.edu/pubs/glove.pdf) which was a major inspiration for me.)

Building on this, a next objective could develop a vector representation of entire sentences. Towards this I wanted to construct a model that would determine if sentences are equivalent.For example, we could work to determine if "The sky is blue" is equivalent to "The color of the sky is blue". I started with simple one or two sentences quora questions. The data consists of pairs of questions generously made available by [data.quora.com](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)

### Applications

The practical use case is to streamline customer support by aggregating customer requests for help. With this model, a business can better assist customers by quickly matching customerquestions to a list of pre-answered questions.

This model could also be a valuable tool for message board sites. Deploy it to compare new posts to existing posts and determine if an equivalent post has been made. This aggregates responses in one place for a better user experience and to facilitate a deeper understanding of what people are discussing on your platform.

### Dataset

The dataset consists of over 400,000 lines of potential question duplicate pairs. Each line contains IDs for each question in the pair, the full text for each question, and a binary value that indicates whether the line truly contains a duplicate pair. Here are a few sample lines of the dataset:

| Question1 | Question2 | Equivalent?
|----------|----------|----------|
|What are natural numbers?  | What is a least natural number? | No
|Which Pizzas are the most popularity ordered pizzas on Domino's menu? | How many calories does a Dominos pizza have? | No
|How do you start a bakery? | How can one start a bakery business? | Yes
|Should I learn python or Java First? | If I had to choose between learning Java and Python, what should I choose to learn first? | Yes

It’s important to recognize a few constraints about this dataset:

- The distribution of questions in the dataset should not be taken to be representative of the distribution of questions asked on Quora. This is, in part, because of the combination of sampling procedures and also due to some sanitization measures that have been applied to the final dataset (e.g., removal of questions with extremely long question details).

- The ground-truth labels contain some amount of noise: they are not guaranteed to be perfect.

### Data Preparation

The data was provided in a clean format. It was taken from a csv into pandas dataframe. The 3 of 400,0000 pairs had null values and were dropped. Punctuation was removed and letters converted to lowercase.

### Neural Networks

#### Word Encoding

The model with the best performance used pre-trained word vectors. This data is made available under the Public Domain Dedication and License v1.0 whose full text can be found at: http://www.opendatacommons.org/licenses/pddl/1.0/. Specifically the 300 dimensional space vectors.

#### Conjoined Networks

![Typical Conjoined Network](https://github.com/DamielCowen/semantic-equivalence/blob/master/src/Conjoined_model2.png "Logo Title Text 1")

The most successful models were based on what I call joincoined neural networks (the literature calls siamese neural networks). These networks have identical encoding layers and identical neural networks. The separate paths are conjoined and passed through a layer with distance metric calculations. The tensors are then passed through a dense and drop out layers before output.  

Conjoined model2 keras deployed gated recurrent unit.

### Cosine Similarity Modeling

Modeling options include options for vectorization and distance calculations. Vectorization are built on top of Sklearns TFIDF Vectorization and Count Vectorization. Distance options are cosine similarity, jaccard distance and euclidean distance.


### Evaluation

The current best model is the conjoined_model2 preforming at a 91% accuracy with Area under ROC of 90%

This model can be accessed [from my googledrive](https://drive.google.com/file/d/1DYECLvdwC123LthIj0lHL-KjddCuEnKG/view?usp=sharing)
