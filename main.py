import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from termcolor import colored, cprint
import numpy as np
import pickle
import base64
from PIL import Image

# loading
with open('tokenizer.pickle2', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_sequence_len = 40
new_model = tf.keras.models.load_model('my_model.h5')


st.write('type your massage ')


def output_2(text, num_of_words):
    seed_text = text
    next_words = num_of_words

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        print(token_list)
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(new_model.predict(token_list), axis=-1)
        # model.predict_classes(token_list, verbose=0)
        cprint(predicted, 'red')
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)
    return seed_text


text = st.text_input(" ", " Type Here")

# level = how many words you want to recomand
level = 1

# output = output_2(text,level)
# st.write(output)


if text:
    next_word = output_2(text, level)

    # Display the predicted next word(s) above the input box
    st.markdown(f"Possible next word: `{next_word}`")

