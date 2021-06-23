from keras.constraints import maxnorm, Constraint
from keras.layers import Input, Masking, GRU, Dense, MaxPooling1D,TimeDistributed, Activation, Dropout,Lambda,Conv1D,Flatten,Conv2D
from keras.models import Model , Sequential
from keras.layers import concatenate,Bidirectional , Reshape
from keras.layers.merge import Dot
from keras.initializers import RandomUniform, RandomNormal
from keras.optimizers import RMSprop, Adam, SGD,Adadelta
from keras.regularizers import l2
import keras.backend as K

import numpy as np


def neuroknapsack(hidden,cnn = False,encoder_gru = False,mem_layers = 1 , dropout_memory = 0.0,dropout_encoder=0.0,dropout_decoder=0.0):
    encoder_inputs = Input(shape=(None, 3))
    #inputs_masking = Masking(mask_value=-1.0)
    #memory-construction
    if cnn:
        sm = Conv1D(hidden,3, activation='relu')(encoder_inputs)
        sm = Dropout(dropout_memory)(sm)
        for _ in range(mem_layers-1):
        	sm = Conv1D(hidden,3, activation='relu')(sm)
        	sm = Dropout(dropout_memory)(sm)
    elif encoder_gru:
        sm = GRU(hidden, dropout=dropout_memory,
                 kernel_initializer=RandomUniform(-0.08, 0.08), return_sequences=True)(encoder_inputs)
        for _ in range(mem_layers-1):
            sm = GRU(hidden, dropout=dropout_memory,
                     kernel_initializer=RandomUniform(-0.08, 0.08),return_sequences=True)(sm)

    else:
        sm = TimeDistributed(Dense(hidden,activation='relu' ,use_bias=False))(encoder_inputs)
        sm = Dropout(dropout_memory)(sm)
        for _ in range(mem_layers-1):
            sm = TimeDistributed(Dense(hidden,activation='relu',use_bias=False))(sm)
            sm = Dropout(dropout_memory)(sm)

    #encoder,
    encoder , state_1 = GRU(hidden, dropout=dropout_encoder,
             kernel_initializer=RandomUniform(-0.08, 0.08), return_state=True)(encoder_inputs)

    #decoder
    #decoder_inputs = Input(shape=(None, 3))
    decoder_tm1 =  Input(shape=(None,None, 2))
    #time-seeker
    #decoder_tm1_masked = Masking(mask_value=0.)(decoder_tm1)
    decoder_tm1_m = TimeDistributed(GRU(hidden,
             kernel_initializer=RandomUniform(-0.08, 0.08),return_sequences=True),name='time-seeker1')(decoder_tm1)
    decoder_tm1_f = TimeDistributed(GRU(hidden,
             kernel_initializer=RandomUniform(-0.08, 0.08),return_sequences=False),name='time-seeker2')(decoder_tm1_m)
    decoder_inputs_f = concatenate([encoder_inputs, decoder_tm1_f])
    sim = Dot(-1,normalize = True, name='cos_sim')

    #sim = TimeDistributed(Dense(hidden,activation='softmax'))

    read_weights = Activation('softmax')

    read_vector = Lambda(lambda x: K.batch_dot(x[0],x[1]),name='read_vector')
    scalar_mul = Lambda(lambda x: x[0]* x[1],name='scalar_mul')
    k_layer = TimeDistributed(Dense(1,activation='relu'))
    decoder_gru = GRU(hidden,
             kernel_initializer=RandomUniform(-0.08, 0.08),return_sequences=True,return_state=True)
    decoder_gru_1 = GRU(hidden,
             kernel_initializer=RandomUniform(-0.08, 0.08),return_sequences=True)
    decoder_gru_2 = GRU(hidden,
             kernel_initializer=RandomUniform(-0.08, 0.08),return_sequences=True)
    decoder_gru_3 = GRU(hidden,dropout=dropout_decoder,
             kernel_initializer=RandomUniform(-0.08, 0.08),return_sequences=True)
    decoder_dense = Dense(2, activation='softmax', name='output')
    mem = Dense(100, activation='softmax',use_bias=False, trainable = False, name='mem')
    memory = Dense(hidden, use_bias=False, trainable = False, name='Memory')

    decoder_outputs,_ = decoder_gru(decoder_inputs_f,initial_state=state_1)
    #decoder 1
    #decoder_outputs  = decoder_gru_1(decoder_outputs)
    #decoder 2
    #decoder_outputs  = decoder_gru_2(decoder_outputs)
    #decoder 3
    #decoder_outputs  = decoder_gru_3(decoder_outputs)
    sim_score = sim([decoder_outputs, sm])
    weights = read_weights(sim_score)
    rt = read_vector([weights,sm])


    #weightsl = mem(decoder_outputs)
    #rl = memory(weightsl)





    decoder_outputs = decoder_dense(concatenate([decoder_outputs,rt]))
    model = Model([encoder_inputs, decoder_tm1], [decoder_outputs])
    model_cos = Model([encoder_inputs, decoder_tm1], [weights])


    #encoder model
    encoder_model = Model(encoder_inputs, [state_1,sm])

    #decoder model
    decoder_state_input = Input(shape=(hidden,))
    sm_input = Input(shape=(None,hidden))
    decoder_outputs, state_h  = decoder_gru(decoder_inputs_f, initial_state=decoder_state_input)
    #decoder 1
    #decoder_outputs  = decoder_gru_1(decoder_outputs)

    #decoder 2
    #decoder_outputs  = decoder_gru_2(decoder_outputs)

    #decoder 3
    #decoder_outputs  = decoder_gru_3(decoder_outputs)
    #k = k_layer(decoder_outputs)
    sim_score = sim([decoder_outputs, sm_input])

    #sim_score = scalar_mul([k,sim_score])
    weights = read_weights(sim_score)
    rt = read_vector([weights,sm_input])
    #weightsl = mem(decoder_outputs)
    #rl = memory(weightsl)

    decoder_outputs = decoder_dense(concatenate([decoder_outputs,rt]))
    decoder_model = Model(
        [sm_input,encoder_inputs,decoder_tm1] + [decoder_state_input],
        [decoder_outputs] + [state_h,rt])
    optimizer = Adam(lr=4e-3,clipnorm=1.0)
    model.compile(optimizer=optimizer,
                    loss={'output':'binary_crossentropy'},
                    metrics={'output':'accuracy'})
    return model , encoder_model , decoder_model ,model_cos
def seq2seq(hidden,dropout_encoder=0.0,dropout_decoder=0.0):
    encoder_inputs = Input(shape=(None, 3))
    #inputs_masking = Masking(mask_value=-1.0)


    #encoder,
    encoder  = GRU(hidden, dropout=dropout_encoder,
             kernel_initializer=RandomUniform(-0.08, 0.08), return_sequences=True)(encoder_inputs)
    encoder , state_1 = GRU(hidden, dropout=dropout_encoder,
             kernel_initializer=RandomUniform(-0.08, 0.08), return_state=True)(encoder)

    #decoder
    decoder_inputs = Input(shape=(None, 3))
    decoder_tm1 =  Input(shape=(None,2))
    decoder_inputs_f = concatenate([decoder_inputs, decoder_tm1])
    decoder_gru = GRU(hidden,  dropout=dropout_decoder,
             kernel_initializer=RandomUniform(-0.08, 0.08),return_sequences=True,return_state=True)
    decoder_gru_1 = GRU(hidden,  dropout=dropout_decoder,
             kernel_initializer=RandomUniform(-0.08, 0.08),return_sequences=True)
    decoder_gru_2 = GRU(hidden,  dropout=dropout_decoder,
             kernel_initializer=RandomUniform(-0.08, 0.08),return_sequences=True)
    decoder_gru_3 = GRU(hidden,  dropout=dropout_decoder,
             kernel_initializer=RandomUniform(-0.08, 0.08),return_sequences=True)
    decoder_dense = Dense(2, activation='softmax', name='output')

    decoder_outputs,_ = decoder_gru(decoder_inputs_f,initial_state=state_1)
    #decoder 1
    decoder_outputs  = decoder_gru_1(decoder_outputs)
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs,decoder_inputs, decoder_tm1], [decoder_outputs])



    #encoder model
    encoder_model = Model(encoder_inputs, [state_1])

    #decoder model
    decoder_state_input = Input(shape=(hidden,))
    sm_input = Input(shape=(None,hidden))
    decoder_outputs, state_h  = decoder_gru(decoder_inputs_f, initial_state=decoder_state_input)
    #decoder 1
    decoder_outputs  = decoder_gru_1(decoder_outputs)
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs,decoder_tm1] + [decoder_state_input],
        [decoder_outputs] + [state_h])
    optimizer = Adam(lr=4e-3,clipnorm=1.0)
    model.compile(optimizer=optimizer,
                    loss={'output':'binary_crossentropy'},
                    metrics={'output':'accuracy'})
    return model , encoder_model , decoder_model
def attentionmodel():
    pass

def grumodel(hidden,dropout_encoder=0.0):
    model = Sequential()
    model.add(GRU(hidden,input_shape=(None,3), dropout=dropout_encoder,return_sequences=True))
    model.add(GRU(hidden,kernel_initializer=RandomUniform(-0.08, 0.08),dropout=dropout_encoder, return_sequences=True))
    model.add(GRU(hidden,kernel_initializer=RandomUniform(-0.08, 0.08),dropout=dropout_encoder, return_sequences=True))
    model.add(Dense(2, activation='softmax'))
    optimizer = Adam(lr=4e-3,clipnorm=1.0)
    model.compile(optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
    return model
def decode_sequence(input_seq,MAX_N,encoder,decoder):
    # Encode the input as state vectors.
    #inputs = np.zeros((1,1,3))
    #inputs [:,:,0] = input_seq[0,0,0]
    #inputs [:,:,1] = input_seq[0,0,1]
    states , sm = encoder.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1,MAX_N, 2))
    #output_tokens = np.zeros((1, 1, 2))
    input_decoder_seq = np.zeros((1,1, 3))
    #rt_input = np.zeros((1, 1, hidden))
    # Populate the first character of target sequence with the start character.
    #target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    y_p = np.zeros((MAX_N,2))
    #rt_ = np.zeros((N,hidden))
    _i = 0
    #c = 0
    #cap = input_seq[0,_i,2] * range
    while not stop_condition:
        input_decoder_seq[0,0] = input_seq[0,_i]
        output_tokens, states, rt = decoder.predict(
            [sm,input_decoder_seq,target_seq] + [states])


        # Sample a token
        _index = np.argmax(output_tokens[0, -1, :])
        #c+= input_seq[0,_i,1] * range

        # Exit condition: either hit max length
        # or find stop character.
        #if _i <= N-2:

        #    rt_input[0, 0,:] = rt
        #    rt_[_i+1] = rt
        if _i >= MAX_N-1:
            stop_condition = True
        else:
            target_seq[0, 0, _i,_index] = 1.
        # Update the target sequence (of length 1).


        #target_seq = np.zeros((1, 1, 2))

        #target_seq[0, 0, _index] = 1.

        y_p[_i,_index] = 1.
        _i+=1
        # Update states
        #states_value = [s1,s2]

    return y_p, rt
def decode_sequence_teacher(input_seq,y,MAX_N,encoder,decoder):
    # Encode the input as state vectors.
    states , sm = encoder.predict([input_seq,y])
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1,MAX_N, 2))
    #output_tokens = np.zeros((1, 1, 2))
    input_decoder_seq = np.zeros((1,1, 3))
    #rt_input = np.zeros((1, 1, hidden))
    # Populate the first character of target sequence with the start character.
    #target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    y_p = np.zeros((MAX_N,2))
    #rt_ = np.zeros((N,hidden))
    _i = 0
    #c = 0
    #cap = input_seq[0,_i,2] * range
    while not stop_condition:
        input_decoder_seq[0,0] = input_seq[0,_i]
        output_tokens, states, rt = decoder.predict(
            [sm,input_decoder_seq,target_seq] + [states])


        # Sample a token
        _index = np.argmax(output_tokens[0, -1, :])
        #c+= input_seq[0,_i,1] * range

        # Exit condition: either hit max length
        # or find stop character.
        #if _i <= N-2:

        #    rt_input[0, 0,:] = rt
        #    rt_[_i+1] = rt
        if _i >= MAX_N-1:
            stop_condition = True
        else:
            target_seq[0, 0, _i,_index] = 1.
        # Update the target sequence (of length 1).


        #target_seq = np.zeros((1, 1, 2))

        #target_seq[0, 0, _index] = 1.

        y_p[_i,_index] = 1.
        _i+=1
        # Update states
        #states_value = [s1,s2]

    return y_p, rt
def decode_sequence_seq2seq(input_seq,MAX_N,encoder,decoder):
    # Encode the input as state vectors.
    states  = encoder.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, 2))
    #output_tokens = np.zeros((1, 1, 2))
    input_decoder_seq = np.zeros((1,1, 3))
    #rt_input = np.zeros((1, 1, hidden))
    # Populate the first character of target sequence with the start character.
    #target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    y_p = np.zeros((MAX_N,2))
    #rt_ = np.zeros((N,hidden))
    _i = 0
    #c = 0
    #cap = input_seq[0,_i,2] * range
    while not stop_condition:
        input_decoder_seq[0,0] = input_seq[0,_i]
        output_tokens, states = decoder.predict(
            [input_decoder_seq,target_seq] + [states])


        # Sample a token
        _index = np.argmax(output_tokens[0, -1, :])
        #c+= input_seq[0,_i,1] * range

        # Exit condition: either hit max length
        # or find stop character.
        #if _i <= N-2:

        #    rt_input[0, 0,:] = rt
        #    rt_[_i+1] = rt
        if _i >= MAX_N-1:
            stop_condition = True
        # Update the target sequence (of length 1).



        target_seq[0, 0, _index] = 1.
        #target_seq[0, 0, _index] = 1.

        y_p[_i] = target_seq[0]
        target_seq = np.zeros((1, 1, 2))
        _i+=1
        # Update states
        #states_value = [s1,s2]

    return y_p
