


################################## GENERAL CHECKPOINTING STRATEGY ################

if False:

    # Checkpoint the weights when validation accuracy improves
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import ModelCheckpoint
    import matplotlib.pyplot as plt
    import numpy
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # load pima indians dataset
    dataset = numpy.loadtxt("pima.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint
    filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)


######################### CHECK-POINT THE BEST MODEL ######################################

if True:

    #Checkpoint the weights for best model on validation accuracy
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import ModelCheckpoint
    import matplotlib.pyplot as plt
    import numpy
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # load pima indians dataset
    dataset = numpy.loadtxt("pima.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))


    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # *************** checkpoint file ****************
    filepath="weights.best.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)


###################################### LOAD WEIGHTS FROM PREVIOUS RUN ###############################

if True:

    # How to load and use weights from a checkpoint
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.callbacks import ModelCheckpoint
    import matplotlib.pyplot as plt
    import numpy
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    # ******************* create model: need to match model that produced checkpoints **************
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

    # ******** load weights from checkpoiter file *******
    model.load_weights("weights.best.hdf5")


    # Compile model (required to make predictions)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print("Created model and loaded weights from file")
    # load pima indians dataset

    dataset = numpy.loadtxt("pima.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]

    # estimate accuracy on whole dataset using loaded weights
    scores = model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    #NOTE: that we never refitted the model in this step
    #      fit was done in the step that created the checkpointer weights


tf.Session().close()
