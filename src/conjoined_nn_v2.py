#3-6-2020 0955

import tensorflow as tf
keras = tf.keras




from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import one_hot, Tokenizer
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, LSTM, Flatten, Dense, Concatenate, Multiply, Dropout, Subtract, Add, Embedding, Activation, GRU
from keras.layers.core import Lambda
from keras import backend as K
from keras.optimizers import Adam, RMSprop



from sklearn.model_selection import train_test_split

from pipeline import vector_comparison

import seaborn as sns


class preprocessing():
    
    def __init__(self):
        print('-/|\-')
    
    def load_data(self, path='../data/training.csv'):
        '''
        Loads data into pandas dataframe. Stores dataframe for access during
        model preprocessing
        '''
        self.df = pd.read_csv(path, index_col = 0)
        
    def split_data(self, test_size = 0.95):
        '''
        Splits the data for training. Saves training and validation data to variabels. 
        
        Generates a corpus of all words used in the data set and creates
        '''
        
        self.pre_train = vector_comparison(self.df)
        self.X_train, self.X_test = self.pre_train.split_data(['question1','question2'], 'is_duplicate') # ran to get self.fitting_text

        self.y_train = self.pre_train.y_train
        self.y_val = self.pre_train.y_test

        
    def tokenize_seqs(self,num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token='<unk>', document_count=0):
        '''
        tokenizes sequiences using keras tokenizer api. 

        returns tokenizer for use in model
        '''

        
        self.token = Tokenizer(num_words=num_words, filters=filters, lower=lower, split=split, char_level=char_level, oov_token=oov_token, document_count=document_count)
        self.token.fit_on_texts(self.pre_train.fitting_text)
        self.seq1 = self.token.texts_to_sequences(self.X_train['question1'].values)
        self.seq2 = self.token.texts_to_sequences(self.X_train['question2'].values)

        #validation data
        self.seq1_val = self.token.texts_to_sequences(self.X_test['question1'].values)
        self.seq2_val = self.token.texts_to_sequences(self.X_test['question2'].values)

        self.vocab_size = len(self.token.index_word) + 1
        return self.token
    



    def pad_sequences(self, max_len):
        '''
        pre-pads sequences.
        
        prerequisite: tokenize_seq
       
        inputs: max_len
        creates: creates lists of padded sequences for seq1_train, seq2_train, seq1_val, seq2_val.
        
        '''

        self.seq1_pad = pad_sequences(self.seq1, maxlen = max_len, padding = 'pre')
        self.seq2_pad = pad_sequences(self.seq2, maxlen = max_len, padding = 'pre')

        self.seq1_pad_val = pad_sequences(self.seq1_val, maxlen = max_len, padding ='pre')
        self.seq2_pad_val = pad_sequences(self.seq2_val, maxlen = max_len, padding ='pre')

    def prep_demo_questions(self, seq1, seq2, max_len):

        #seq1 = seq1.split()
        #seq2 = seq2.split()
        

        seq1, seq2 = self.demo_pad(seq1, seq2, max_len)

        return seq1, seq2

    def demo_pad(self, seq1, seq2, max_len):
        seq1_pad_needed = max_len - len(list(seq1))
        seq2_pad_needed = max_len - len(list(seq2))
    
        seq1_out = []
        seq2_out = []
    
        for pad in range(seq1_pad_needed):
            seq1_out.append(0)
        for value in seq1:
            seq1_out.append(value)
            

        for pad in range(seq2_pad_needed):
            seq2_out.append(0)
        for value in seq2:
            seq2_out.append(value)
        
        return seq1_out, seq2_out
    
                          

    


    
class conjoined():

    def __init__(self, token):
        self.embedding_index = dict()
        self.token = token
        self.vocab_size = len(token.index_word) + 1
        #self.embedding_matrix = np.zeros((self.vocab_size, self.embeded_dim))
    
        print(' | 3 |')
        print('-0 | 0-')
        print(' | | |')
        print('-0 | 0-')
        print(' 0   0 ')
        print('  \ / ')


    def load_word_encoder(self, path = 'src/pre_trained/glove.6B/glove.6B.300d.txt'):
        
        f = open(path)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embedding_index[word] = coefs
        f.close()

        self.embeded_dim = len(self.embedding_index['the'])
        self.weight_shape = self.embedding_index['the'].shape[0]
        self.embedding_matrix = np.zeros((self.vocab_size, self.embeded_dim))

        print('Loaded %s word vectors.' % len(self.embedding_index))



    def create_embeddings_matrix(self):
        '''
        creates a nxm matrix where n are the words modein vocabulary and m are the vectors from encoders. 
        if vocab word not in pretrained encoder masked with <unkw>

        '''
        print('create_embeddings_matrix')
        #word_index = token.word_index
        #self.embedding_matrix = np.zeros((self.vocab_size, self.embeded_dim))
        for word, i in self.token.word_index.items():
            
            embeddings_vector = self.embedding_index.get(word)
            #print(word, i, embeddings_vector is not None)
            if embeddings_vector is not None:
                self.embedding_matrix[i] = embeddings_vector

    def cosine_distance(self,vests):
        '''
        computes cosine_distance in tensor space.
        
        inputs: 2 tensors of equal dimensions
        outputs: 1 tensor layer which computes cosine distances
        '''
        x, y = vests
        x = K.l2_normalize(x, axis=-1)
        y = K.l2_normalize(y, axis=-1)
        return -K.mean(x * y, axis=-1, keepdims=True)

    def cos_dist_output_shape(self, shapes):
        '''
        outputs the shape of the cosine distance tensor.
        
        '''
        shape1, shape2 = shapes
        return (shape1[0],1)

    
    def compile_model_1(self, input1, input2, max_len):
        '''
        The model Sequences are passed through the common embedder and than two identical LSTM layers before concatenation. 
        The concatination layer contains cosine distance, absolute square distance and 


        '''

        input_1 = Input(shape=(input1.shape[1],))
        input_2 = Input(shape=(input2.shape[1],))

        
        model = Sequential()
        e = Embedding(self.vocab_size, self.weight_shape, weights=[self.embedding_matrix], input_length=max_len, trainable=False)

        common_lstm_1 = LSTM(32,return_sequences=True, activation="relu")
        common_lstm_2 = LSTM(32,return_sequences=True, activation="relu")


        lstm_1 = e(input_1)
        lstm_2 = e(input_2)

        #seq1 path
        vector_1 = common_lstm_1(lstm_1)
        vector_1 = common_lstm_2(vector_1)
        vector_1 = Flatten()(vector_1)

        #seq2 path
        vector_2 = common_lstm_1(lstm_2)
        vector_2 = common_lstm_2(vector_2)
        vector_2 = Flatten()(vector_2)


        x3 = Subtract()([vector_1, vector_2])
        x3 = Multiply()([x3, x3])

        x1_ = Multiply()([vector_1, vector_1])
        x2_ = Multiply()([vector_2, vector_2])
        x4 = Subtract()([x1_, x2_])



        cosine_sim = Lambda(self.cosine_distance, output_shape=self.cos_dist_output_shape)([vector_1, vector_2])

        conc = Concatenate(axis=-1)([cosine_sim,x4, x3])


        x = Dense(100, activation="relu", name='conc_layer')(conc)
        x = Dropout(0.01)(x)
        out = Dense(1, activation="relu", name = 'out')(x)

        model = Model([input_1, input_2], out)

        model.compile(loss="binary_crossentropy", metrics=['acc',tf.keras.metrics.AUC()], optimizer=Adam(0.0001))

        self.model1 =  model

        return model
    
    
    def compile_model_2(self, input1, input2, max_len):
        
        input_1 = Input(shape=(input1.shape[1],))
        input_2 = Input(shape=(input2.shape[1],))

        
        
        model2 = Sequential()
        
        common_GRU = GRU(32,return_sequences=True, activation="relu")
        e = Embedding(self.vocab_size, self.weight_shape, weights=[self.embedding_matrix], input_length=max_len,
                      trainable=False)
       

        #seq1
        vector_1 = e(input_1)
        vector_1 = common_GRU(vector_1)
        vector_1 = Flatten()(vector_1)

        #seq2
        vector_2 = e(input_2)
        vector_2 = common_GRU(vector_2)
        vector_2 = Flatten()(vector_2)
        
        #distance option 1
        x3 = Subtract()([vector_1, vector_2])
        x3 = Multiply()([x3, x3]) 

        
        #distance option 2
        x1_ = Multiply()([vector_1, vector_1])
        x2_ = Multiply()([vector_2, vector_2])
        x4 = Subtract()([x1_, x2_]) #distance option 2

        #distance option3
        cosine_sim = Lambda(self.cosine_distance, output_shape=self.cos_dist_output_shape)([vector_1, vector_2])
        
        #distance concatination layer,allows model to learn best option.
        conc = Concatenate(axis=-1)([cosine_sim,x4, x3])


        x = Dense(100, activation="relu", name='conc_layer')(conc)
        x = Dropout(0.01)(x)
        out = Dense(1, activation="sigmoid", name = 'out')(x)

        model2 = Model([input_1, input_2], out)

        model2.compile(loss="binary_crossentropy", metrics=['acc',tf.keras.metrics.AUC()], optimizer=Adam(0.01))
        self.model2 =  model2

        return model2



if __name__ == "__main__":
    # execute only if run as a script

    print('initializing preprocessing')
    pre = preprocessing()
    #load question pairs data
    path = '../Data/training.csv'
    pre.load_data(path)
    print('data loaded')

    #split data
    pre.split_data()
    print('data split')

    #tokenizes sequences see sample in cell below 
    token = pre.tokenize_seqs()
    print('tokenize sequencies')
    max_len = 30
    #pad sequences
    pre.pad_sequences(max_len)
    print('the sequences have been padded')
    

    print('initializing conjoined network')
    model = conjoined(token)

    print('loading encoder')
    encoder_path = 'pre_trained/glove.6B/glove.6B.300d.txt'
    model.load_word_encoder(encoder_path)

    model.create_embeddings_matrix()



    model2 = model.compile_model_2(pre.seq1_pad, pre.seq2_pad, max_len = max_len )
    batchsize = np.sqrt(len(pre.seq1_pad))

    model2.summary()
    

    model2.fit([pre.seq1_pad,pre.seq2_pad],
           pre.y_train.values.reshape(-1,1),
           epochs = 50, 
           batch_size = 500,
           validation_data=(
               [pre.seq1_pad_val,pre.seq2_pad_val],
               pre.y_val.values.reshape(-1,1)))

    save_model_path = '../models/'

    model2.save_weights(f'{save_model_path}model2_weights.h5')


    print('saving model history')
    results_df = pd.DataFrame(model2.history.history)
    results_df.to_csv(f'{save_model_path}model2_results.csv')
    print('model history saved. Good bye!')
