from flask import Flask, render_template,request
import pickle
import dill
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense
from transformers import AutoTokenizer,TFBertModel
from sklearn.metrics import classification_report
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained('bert-base-cased')

tokenizer.save_pretrained('bert-tokenizer')
bert.save_pretrained('bert-model')
import shutil
shutil.make_archive('bert-tokenizer', 'zip', 'bert-tokenizer')
shutil.make_archive('bert-model','zip','bert-model')

from transformers import BertTokenizer, TFBertModel, BertConfig,TFDistilBertModel,DistilBertTokenizer,DistilBertConfig
dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
# with open('model.pkl', 'rb') as in_strm:
#     model = dill.load(in_strm)
#h=pickle.load(open('./model.pkl','rb'))

max_len = 70


input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
# embeddings = dbert_model(input_ids,attention_mask = input_mask)[0]


embeddings = bert(input_ids,attention_mask = input_mask)[0] #(0 is the last hidden states,1 means pooler_output)
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)

y = Dense(6,activation = 'sigmoid')(out)
    
new_model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
new_model.layers[2].trainable = True
# for training bert our lr must be so small

new_model.load_weights('sentiment_weights.h5')

dict=['anger','fear','joy','love','sadness','surprise']
app=Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('firebase2.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_text=request.form.get('user-input')
    x_val = tokenizer(
    text=user_text,
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding='max_length', 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True) 
    validation = new_model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
    #print(np.argmax(validation))
    for i in range(validation[0].shape[0]):
        print(dict[i], validation[0][i])

    print(f"Your current Mood: {dict[np.argmax(validation)]}")    
    return "success"    




if __name__ == '__main__':
    app.run(port=3000, debug=True)