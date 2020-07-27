import numpy as np
import pandas as pd
from keras.layers import *
from keras.callbacks import *
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import Model,Sequential
from keras.activations import softmax,sigmoid
from keras.optimizers import Adam
from keras import losses
import codecs
import yaml
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras_self_attention import SeqSelfAttention,SeqWeightedAttention
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint
from keras_xlnet import ATTENTION_TYPE_UNI

## 参数
maxlen = 100
Batch_size=16
Epoch =1

config_path="/home/thefair/haoyi/ad_classification/xlnet_cased_L-12_H-768_A-12/xlnet_config.json"
checkpoint_path="/home/thefair/haoyi/ad_classification/xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt"
vocab_path="/home/thefair/haoyi/ad_classification/xlnet_cased_L-12_H-768_A-12/spiece.model"
def get_token_dict(dict_path):
    """

    :param dict_path:bert模型的vocab.txt文件
    :return: 将文件中字进行编码
    """
    token_dict={}
    with codecs.open(dict_path,"r","utf-8") as reader:
        for line in reader:
            token=line.strip()
            token_dict[token]=len(token_dict)
    return token_dict

def data_training_generator():
    """
    读取正负样本，并生成训练集合和测试集合
    :return: training dataset validation dataset
    """
    neg = []
    pos = []
    with codecs.open("positive_content.txt","r",'utf-8') as reader:
        for line in reader:
            pos.append(line.strip())

    with codecs.open("negative_content.txt","r","utf-8") as reader:
        for line in reader:
            neg.append(line.strip())

    return pos[:9000],neg[:9000]

def get_encode(pos,neg):
    """

    :param pos:正样本
    :param neg:负样本
    :return:
    """
    all_data=pos+neg
    X=[]
    tokenizer=Tokenizer(vocab_path)
    for line in all_data:
        tokens=tokenizer.encode(line)
        X.append(tokens)
    X=sequence.pad_sequences(X,maxlen=maxlen,padding='post',truncating='post')
    return X

def build_xlnet_model(X):
    '''
    构建xlnet模型
    :param X: encode后的结果
    :return: np array形式做训练
    '''
    XLNetmodel = load_trained_model_from_checkpoint(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        batch_size=16,
        memory_len=0,
        target_len=100,
        in_train_phase=False,
        attention_type=ATTENTION_TYPE_UNI,
    )
    memory_length_input = np.zeros((1, 1))
    wordvec=[]
    for line in X:
        token_input = np.expand_dims(np.array(line), axis=0)
        segment_input = np.zeros_like(token_input)
        XLNet_output=XLNetmodel.predict_on_batch([token_input, segment_input, memory_length_input])
        wordvec.append(XLNet_output[0])
    return np.array(wordvec)

def build_model_attention():
    model=Sequential()
    model.add(Bidirectional(LSTM(units=128,dropout=0.5,recurrent_dropout=0.5,return_sequences=True)))
    #model.add(Bidirectional(LSTM(128,recurrent_dropout=0.5)))
    model.add(SeqWeightedAttention())
    model.add(Dense(1,activation=sigmoid))
    model.compile(loss=losses.binary_crossentropy,optimizer=Adam(1e-5),metrics=['accuracy'])
    return model

def build_model():
    model=Sequential()
    model.add(Bidirectional(LSTM(128,return_sequences=True)))
    model.add(Dense(1,activation=sigmoid))
    model.compile(loss=losses.binary_crossentropy,optimizer=Adam(1e-5),metrics=['categorical_accuracy'])
    return model

def train(wordvec,y):
    model=build_model_attention()
    early_stopping=EarlyStopping(monitor='val_loss',patience=20,verbose=2)
    filepath="weights-title-attention2_best.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True,mode='max')
    model.fit(wordvec,y,batch_size=32,epochs=100,validation_split=0.2,callbacks=[early_stopping])
    #yaml_string=model.to_yaml()
    #with open("test_keras_bert3.yml","w") as f:
    #    f.write(yaml.dump(yaml_string,default_flow_style=True))
    model.save('test_keras_bert5.h5')


if __name__=="__main__":
    print("开始")
    pos,neg=data_training_generator()
    print("开始 encode")
    X1=get_encode(pos,neg)
    print("开始 XLNet")
    wordvec=build_xlnet_model(X1)
    y=np.concatenate((np.ones(9000,dtype=int),np.zeros(9000,dtype=int)))
    print("开始训练")
    train(wordvec,y)