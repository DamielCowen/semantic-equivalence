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


### Modeling


### Evaluation


### Deployment

