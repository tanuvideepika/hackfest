
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)






import warnings
warnings.filterwarnings('ignore')
%config Completer.use_jedi = False # if autocompletion doesnot work in kaggle notebook | hit tab


# importing the dataset 
df_train = pd.read_csv('../input/train.txt', header =None, sep =';', names = ['Input','Sentiment'], encoding='utf-8')
df_test = pd.read_csv('../input/test.txt', header = None, sep =';', names = ['Input','Sentiment'],encoding='utf-8')
df_val=pd.read_csv('../input/val.txt',header=None,sep=';',names=['Input','Sentiment'],encoding='utf-8')

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:19.110979Z","iopub.execute_input":"2023-03-28T18:48:19.111565Z","iopub.status.idle":"2023-03-28T18:48:19.125720Z","shell.execute_reply.started":"2023-03-28T18:48:19.111527Z","shell.execute_reply":"2023-03-28T18:48:19.124761Z"}}
df_full = pd.concat([df_train,df_test,df_val], axis = 0)
df_full

# %% [markdown] {"editable":false}
# #### here we are doing some text preprocessing 
# 
# 

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:19.127042Z","iopub.execute_input":"2023-03-28T18:48:19.127381Z","iopub.status.idle":"2023-03-28T18:48:25.077430Z","shell.execute_reply.started":"2023-03-28T18:48:19.127346Z","shell.execute_reply":"2023-03-28T18:48:25.076366Z"}}
!pip install text_hammer

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:25.079939Z","iopub.execute_input":"2023-03-28T18:48:25.080330Z","iopub.status.idle":"2023-03-28T18:48:25.086854Z","shell.execute_reply.started":"2023-03-28T18:48:25.080287Z","shell.execute_reply":"2023-03-28T18:48:25.086044Z"}}
import text_hammer as th

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:25.088244Z","iopub.execute_input":"2023-03-28T18:48:25.088679Z","iopub.status.idle":"2023-03-28T18:48:25.103553Z","shell.execute_reply.started":"2023-03-28T18:48:25.088640Z","shell.execute_reply":"2023-03-28T18:48:25.102748Z"}}
%%time

from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()

def text_preprocessing(df,col_name):
    column = col_name
    df[column] = df[column].progress_apply(lambda x:str(x).lower())
    df[column] = df[column].progress_apply(lambda x: th.cont_exp(x)) #you're -> you are; i'm -> i am
    df[column] = df[column].progress_apply(lambda x: th.remove_emails(x))
    df[column] = df[column].progress_apply(lambda x: th.remove_html_tags(x))
#     df[column] = df[column].progress_apply(lambda x: ps.remove_stopwords(x))

    df[column] = df[column].progress_apply(lambda x: th.remove_special_chars(x))
    df[column] = df[column].progress_apply(lambda x: th.remove_accented_chars(x))
#     df[column] = df[column].progress_apply(lambda x: th.make_base(x)) #ran -> run,
    return(df)

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:25.105668Z","iopub.execute_input":"2023-03-28T18:48:25.106304Z","iopub.status.idle":"2023-03-28T18:48:54.198413Z","shell.execute_reply.started":"2023-03-28T18:48:25.106265Z","shell.execute_reply":"2023-03-28T18:48:54.197541Z"}}
df_cleaned = text_preprocessing(df_full,'Input')

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:54.201395Z","iopub.execute_input":"2023-03-28T18:48:54.201953Z","iopub.status.idle":"2023-03-28T18:48:54.206546Z","shell.execute_reply.started":"2023-03-28T18:48:54.201912Z","shell.execute_reply":"2023-03-28T18:48:54.205764Z"}}
df_cleaned = df_cleaned.copy()

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:54.208506Z","iopub.execute_input":"2023-03-28T18:48:54.209043Z","iopub.status.idle":"2023-03-28T18:48:54.248517Z","shell.execute_reply.started":"2023-03-28T18:48:54.209002Z","shell.execute_reply":"2023-03-28T18:48:54.247645Z"}}
df_cleaned['num_words'] = df_cleaned.Input.apply(lambda x:len(x.split()))

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:54.249847Z","iopub.execute_input":"2023-03-28T18:48:54.250194Z","iopub.status.idle":"2023-03-28T18:48:54.261933Z","shell.execute_reply.started":"2023-03-28T18:48:54.250156Z","shell.execute_reply":"2023-03-28T18:48:54.261020Z"}}
# changing the data type to the category to encode into codes 
df_cleaned['Sentiment'] = df_cleaned.Sentiment.astype('category')


# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:54.263241Z","iopub.execute_input":"2023-03-28T18:48:54.263879Z","iopub.status.idle":"2023-03-28T18:48:54.273904Z","shell.execute_reply.started":"2023-03-28T18:48:54.263840Z","shell.execute_reply":"2023-03-28T18:48:54.272945Z"}}
df_cleaned.Sentiment

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:54.275648Z","iopub.execute_input":"2023-03-28T18:48:54.276004Z","iopub.status.idle":"2023-03-28T18:48:54.287079Z","shell.execute_reply.started":"2023-03-28T18:48:54.275969Z","shell.execute_reply":"2023-03-28T18:48:54.285969Z"}}
df_cleaned.Sentiment.cat.codes

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:54.288407Z","iopub.execute_input":"2023-03-28T18:48:54.288767Z","iopub.status.idle":"2023-03-28T18:48:54.297278Z","shell.execute_reply.started":"2023-03-28T18:48:54.288733Z","shell.execute_reply":"2023-03-28T18:48:54.296515Z"}}
encoded_dict  = {'anger':0,'fear':1, 'joy':2, 'love':3, 'sadness':4, 'surprise':5}

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:54.298104Z","iopub.execute_input":"2023-03-28T18:48:54.298600Z","iopub.status.idle":"2023-03-28T18:48:54.313403Z","shell.execute_reply.started":"2023-03-28T18:48:54.298570Z","shell.execute_reply":"2023-03-28T18:48:54.312601Z"}}
df_cleaned['Sentiment']  =  df_cleaned.Sentiment.cat.codes
df_cleaned.Sentiment

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:54.314635Z","iopub.execute_input":"2023-03-28T18:48:54.315159Z","iopub.status.idle":"2023-03-28T18:48:54.324768Z","shell.execute_reply.started":"2023-03-28T18:48:54.315104Z","shell.execute_reply":"2023-03-28T18:48:54.323955Z"}}
df_cleaned.num_words.max()

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:54.326087Z","iopub.execute_input":"2023-03-28T18:48:54.326804Z","iopub.status.idle":"2023-03-28T18:48:54.355102Z","shell.execute_reply.started":"2023-03-28T18:48:54.326762Z","shell.execute_reply":"2023-03-28T18:48:54.354211Z"}}
from sklearn.model_selection import train_test_split
data_train,data_test = train_test_split(df_cleaned, test_size = 0.3, random_state = 42, stratify = df_cleaned.Sentiment)

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:54.359867Z","iopub.execute_input":"2023-03-28T18:48:54.360505Z","iopub.status.idle":"2023-03-28T18:48:54.374289Z","shell.execute_reply.started":"2023-03-28T18:48:54.360434Z","shell.execute_reply":"2023-03-28T18:48:54.373465Z"}}
data_train.shape

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:54.377980Z","iopub.execute_input":"2023-03-28T18:48:54.381409Z","iopub.status.idle":"2023-03-28T18:48:54.386890Z","shell.execute_reply.started":"2023-03-28T18:48:54.381362Z","shell.execute_reply":"2023-03-28T18:48:54.386000Z"}}
data_test.shape

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:54.393886Z","iopub.execute_input":"2023-03-28T18:48:54.394236Z","iopub.status.idle":"2023-03-28T18:48:59.591013Z","shell.execute_reply.started":"2023-03-28T18:48:54.394192Z","shell.execute_reply":"2023-03-28T18:48:59.590089Z"}}
from tensorflow.keras.utils import to_categorical

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:59.593229Z","iopub.execute_input":"2023-03-28T18:48:59.593497Z","iopub.status.idle":"2023-03-28T18:48:59.601738Z","shell.execute_reply.started":"2023-03-28T18:48:59.593469Z","shell.execute_reply":"2023-03-28T18:48:59.600867Z"}}
to_categorical(data_train.Sentiment)

# %% [markdown] {"editable":false}
# ### so far we have cleaned our text data now we need to encode our output in some labels , 
# here we have two method to encode 

# %% [markdown] {"editable":false}
# ## Now lets load the model 

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:48:59.603211Z","iopub.execute_input":"2023-03-28T18:48:59.603788Z","iopub.status.idle":"2023-03-28T18:49:23.870454Z","shell.execute_reply.started":"2023-03-28T18:48:59.603752Z","shell.execute_reply":"2023-03-28T18:49:23.869571Z"}}
from transformers import AutoTokenizer,TFBertModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained('bert-base-cased')


# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:49:23.871962Z","iopub.execute_input":"2023-03-28T18:49:23.872535Z","iopub.status.idle":"2023-03-28T18:49:24.563307Z","shell.execute_reply.started":"2023-03-28T18:49:23.872496Z","shell.execute_reply":"2023-03-28T18:49:24.562386Z"}}
tokenizer.save_pretrained('bert-tokenizer')
bert.save_pretrained('bert-model')
# for saving model locally and we can load it later on 

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:49:24.564646Z","iopub.execute_input":"2023-03-28T18:49:24.565033Z","iopub.status.idle":"2023-03-28T18:49:24.591331Z","shell.execute_reply.started":"2023-03-28T18:49:24.564994Z","shell.execute_reply":"2023-03-28T18:49:24.590325Z"}}
import shutil
shutil.make_archive('bert-tokenizer', 'zip', 'bert-tokenizer')

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:49:24.592639Z","iopub.execute_input":"2023-03-28T18:49:24.592999Z","iopub.status.idle":"2023-03-28T18:49:46.709424Z","shell.execute_reply.started":"2023-03-28T18:49:24.592962Z","shell.execute_reply":"2023-03-28T18:49:46.708594Z"}}
shutil.make_archive('bert-model','zip','bert-model')

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:49:46.710673Z","iopub.execute_input":"2023-03-28T18:49:46.711023Z","iopub.status.idle":"2023-03-28T18:49:54.784171Z","shell.execute_reply.started":"2023-03-28T18:49:46.710985Z","shell.execute_reply":"2023-03-28T18:49:54.783289Z"}}
### we can use distilbert its lighter cheaper and similar performance 

from transformers import BertTokenizer, TFBertModel, BertConfig,TFDistilBertModel,DistilBertTokenizer,DistilBertConfig
dbert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')


# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:49:54.788065Z","iopub.execute_input":"2023-03-28T18:49:54.788329Z","iopub.status.idle":"2023-03-28T18:49:54.797504Z","shell.execute_reply.started":"2023-03-28T18:49:54.788301Z","shell.execute_reply":"2023-03-28T18:49:54.796586Z"}}
tokenizer('hello this me abhishek')

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:49:54.799159Z","iopub.execute_input":"2023-03-28T18:49:54.799517Z","iopub.status.idle":"2023-03-28T18:49:56.690461Z","shell.execute_reply.started":"2023-03-28T18:49:54.799478Z","shell.execute_reply":"2023-03-28T18:49:56.689460Z"}}
# Tokenize the input (takes some time) 
# here tokenizer using from bert-base-cased
x_train = tokenizer(
    text=data_train.Input.tolist(),
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)


x_test = tokenizer(
    text=data_test.Input.tolist(),
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)



# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:49:56.691784Z","iopub.execute_input":"2023-03-28T18:49:56.692115Z","iopub.status.idle":"2023-03-28T18:49:56.744457Z","shell.execute_reply.started":"2023-03-28T18:49:56.692077Z","shell.execute_reply":"2023-03-28T18:49:56.743571Z"}}
x_test['input_ids']

# %% [code] {"editable":false}


# %% [markdown] {"editable":false}
# ### loading some libraries 

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:49:56.745808Z","iopub.execute_input":"2023-03-28T18:49:56.746319Z","iopub.status.idle":"2023-03-28T18:49:57.163109Z","shell.execute_reply.started":"2023-03-28T18:49:56.746278Z","shell.execute_reply":"2023-03-28T18:49:57.162120Z"}}
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:49:57.164649Z","iopub.execute_input":"2023-03-28T18:49:57.164968Z","iopub.status.idle":"2023-03-28T18:49:57.178030Z","shell.execute_reply.started":"2023-03-28T18:49:57.164940Z","shell.execute_reply":"2023-03-28T18:49:57.177212Z"}}
import tensorflow as tf
tf.config.experimental.list_physical_devices('GPU')

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:49:57.179384Z","iopub.execute_input":"2023-03-28T18:49:57.179944Z","iopub.status.idle":"2023-03-28T18:50:03.425132Z","shell.execute_reply.started":"2023-03-28T18:49:57.179906Z","shell.execute_reply":"2023-03-28T18:50:03.424266Z"}}
max_len = 70
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense

input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
# embeddings = dbert_model(input_ids,attention_mask = input_mask)[0]


embeddings = bert(input_ids,attention_mask = input_mask)[0] #(0 is the last hidden states,1 means pooler_output)
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)

y = Dense(6,activation = 'sigmoid')(out)
    
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True
# for training bert our lr must be so small

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:50:03.426420Z","iopub.execute_input":"2023-03-28T18:50:03.426776Z","iopub.status.idle":"2023-03-28T18:50:03.451131Z","shell.execute_reply.started":"2023-03-28T18:50:03.426734Z","shell.execute_reply":"2023-03-28T18:50:03.450345Z"}}
optimizer = Adam(
    learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website 
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)

# Set loss and metrics
loss =CategoricalCrossentropy(from_logits = True)
metric = CategoricalAccuracy('balanced_accuracy'),
# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)

# %% [code] {"editable":false}


# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:50:03.452517Z","iopub.execute_input":"2023-03-28T18:50:03.452892Z","iopub.status.idle":"2023-03-28T18:50:03.475435Z","shell.execute_reply.started":"2023-03-28T18:50:03.452856Z","shell.execute_reply":"2023-03-28T18:50:03.474495Z"}}
model.summary()

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:50:03.476681Z","iopub.execute_input":"2023-03-28T18:50:03.477174Z","iopub.status.idle":"2023-03-28T18:50:03.481374Z","shell.execute_reply.started":"2023-03-28T18:50:03.477138Z","shell.execute_reply":"2023-03-28T18:50:03.480394Z"}}
tf.config.experimental_run_functions_eagerly(True)
tf.config.run_functions_eagerly(True)


# %% [markdown] {"editable":false}
# #### model fitting and then evaluation

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:50:03.482550Z","iopub.execute_input":"2023-03-28T18:50:03.483060Z","iopub.status.idle":"2023-03-28T18:53:25.404480Z","shell.execute_reply.started":"2023-03-28T18:50:03.483023Z","shell.execute_reply":"2023-03-28T18:53:25.403589Z"}}
train_history = model.fit(
    x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
    y = to_categorical(data_train.Sentiment),
    validation_data = (
    {'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}, to_categorical(data_test.Sentiment)
    ),
  epochs=1,
    batch_size=36
)

# %% [markdown] {"editable":false}
# ### model.save doesn't work in this case 
# so we need to save the weights files and then we need to make the same model architecture and then load with the weights

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:53:25.405979Z","iopub.execute_input":"2023-03-28T18:53:25.406324Z","iopub.status.idle":"2023-03-28T18:53:26.045841Z","shell.execute_reply.started":"2023-03-28T18:53:25.406286Z","shell.execute_reply":"2023-03-28T18:53:26.044880Z"}}
model.save_weights('sentiment_weights.h5')


# %% [markdown] {"editable":false}
# lets create a new model and then load the weights 

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:53:26.047163Z","iopub.execute_input":"2023-03-28T18:53:26.047492Z","iopub.status.idle":"2023-03-28T18:53:26.051416Z","shell.execute_reply.started":"2023-03-28T18:53:26.047453Z","shell.execute_reply":"2023-03-28T18:53:26.050604Z"}}
# max_len = 70
# import tensorflow as tf
# from tensorflow.keras.layers import Input, Dense

# input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
# input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
# # embeddings = dbert_model(input_ids,attention_mask = input_mask)[0]


# embeddings = bert(input_ids,attention_mask = input_mask)[0] #(0 is the last hidden states,1 means pooler_output)
# out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
# out = Dense(128, activation='relu')(out)
# out = tf.keras.layers.Dropout(0.1)(out)
# out = Dense(32,activation = 'relu')(out)

# y = Dense(6,activation = 'sigmoid')(out)
    
# new_model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
# new_model.layers[2].trainable = True
# # for training bert our lr must be so small

# new_model.load_weights('sentiment_weights.h5')

# %% [markdown] {"editable":false}
# ### Prediction Part

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:53:26.052860Z","iopub.execute_input":"2023-03-28T18:53:26.053444Z","iopub.status.idle":"2023-03-28T18:53:43.294904Z","shell.execute_reply.started":"2023-03-28T18:53:26.053407Z","shell.execute_reply":"2023-03-28T18:53:43.294014Z"}}
predicted_raw = model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:53:43.296315Z","iopub.execute_input":"2023-03-28T18:53:43.296675Z","iopub.status.idle":"2023-03-28T18:53:43.305160Z","shell.execute_reply.started":"2023-03-28T18:53:43.296637Z","shell.execute_reply":"2023-03-28T18:53:43.304232Z"}}
predicted_raw[0]

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:53:43.306872Z","iopub.execute_input":"2023-03-28T18:53:43.307357Z","iopub.status.idle":"2023-03-28T18:53:43.313755Z","shell.execute_reply.started":"2023-03-28T18:53:43.307321Z","shell.execute_reply":"2023-03-28T18:53:43.312995Z"}}
y_predicted = np.argmax(predicted_raw, axis = 1)

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:53:43.314937Z","iopub.execute_input":"2023-03-28T18:53:43.315283Z","iopub.status.idle":"2023-03-28T18:53:43.326137Z","shell.execute_reply.started":"2023-03-28T18:53:43.315246Z","shell.execute_reply":"2023-03-28T18:53:43.325157Z"}}
data_test.Sentiment

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:53:43.327359Z","iopub.execute_input":"2023-03-28T18:53:43.327913Z","iopub.status.idle":"2023-03-28T18:53:43.335992Z","shell.execute_reply.started":"2023-03-28T18:53:43.327867Z","shell.execute_reply":"2023-03-28T18:53:43.335130Z"}}
from sklearn.metrics import classification_report


# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T18:53:43.337194Z","iopub.execute_input":"2023-03-28T18:53:43.337600Z","iopub.status.idle":"2023-03-28T18:53:43.362445Z","shell.execute_reply.started":"2023-03-28T18:53:43.337565Z","shell.execute_reply":"2023-03-28T18:53:43.361502Z"}}
print(classification_report(data_test.Sentiment, y_predicted))

# %% [markdown] {"editable":false}
# ### for prediction lets

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T19:04:02.853952Z","iopub.execute_input":"2023-03-28T19:04:02.854288Z"}}
texts = input(str('input the text'))

x_val = tokenizer(
    text=texts,
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding='max_length', 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True) 
validation = model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
validation

# %% [code] {"editable":false,"execution":{"iopub.status.busy":"2023-03-28T19:03:46.051305Z","iopub.execute_input":"2023-03-28T19:03:46.051644Z","iopub.status.idle":"2023-03-28T19:03:46.059081Z","shell.execute_reply.started":"2023-03-28T19:03:46.051612Z","shell.execute_reply":"2023-03-28T19:03:46.056381Z"}}
for key , value in zip(encoded_dict.keys(),validation[0]):
    print(key,value)

# %% [code]
