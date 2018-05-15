from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

"""Useful links

- Keras
https://keras.io/getting-started/sequential-model-guide/
https://keras.io/layers/recurrent/

- Guide & Tutorials
http://vict0rsch.github.io/tutorials/keras/recurrent/
"""

"""
Suppose we have three actions
- NAVIGATE
- WAITFOR
- FOLLOW

Every action correspond to a subject, because action is a verb and subject is usually the robot, we
care more about the object. In general, the action and subject should be very easy to parse.

Suppose we have a sentence
- "can you navigate to shipping"

The subject is 'you', the verb is 'navigate' and the object is 'shipping'.

Let's denote a couple tokens
- 0 => <NULL>
- 1 => <SUBJECT>
- 2 => <OBJECT>
- 3 => <ACTION:Navigate>
- 4 => <Action:WaitFor>
- 5 => <Action:Follow>

Now we can use many-to-many LSTM to parse the sentence into relevant tokens
- Input => "can you navigate to shipping"
- Output => "0 1 3 3 2"

Using the token, we can create a task that that is assigned to <SUBJECT> with <ACTION:Navigate> 
which will receive inputs of <OBJECT>.
"""


def main():
    sentence_len = 20
    vocab_size = 200
    word_vec_dim = 128
    
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=word_vec_dim))
    model.add(LSTM(units=1, 
                   input_shape=(sentence_len, word_vec_dim), 
                   activation='tanh', 
                   recurrent_activation='hard_sigmoid', 
                   return_sequences=True))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


if __name__ == '__main__':
    main()