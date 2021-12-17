def main_training():

    # loading data

    data, label_file = load_data_augmented('',['data_augmented5'])

    data = np.expand_dims(data,axis=0)

    data = data.reshape(26170,48,800,1)

    data = data/255.


    # loading label and process it

    label = read_final_label('data_augmented5.txt')

    # shuffle
    data, label = shuffle(data,label, random_state=42)

    label = [i.replace('||',' ') for i in label]
    label = [i.replace('|',' ') for i in label]
    label = [i[1:] if i[0]==' ' else i for i in label]
    label = [i[:-1] if i[-1]==' ' else i for i in label]
    label = [i.replace('  ',' ') for i in label]
    label = [i.replace('\n','') for i in label]

    chars = string.printable[:95]

    encoded_label = [encode_string(i,chars) for i in label]

    maxv = 0
    for i in encoded_label:
        if len(i) > maxv:
            maxv = len(i)


    pad_label = copy.deepcopy(encoded_label)
    for i in range(len(pad_label)):
        if len(pad_label[i])<maxv:
            diff = maxv - len(pad_label[i])
            for j in range(diff):
                pad_label[i].append(chars.find(' '))

    # train test split
    valid_padded_txt = list()
    train_padded_txt = list()
    test_padded_txt = list()
    # lists for training dataset
    train_img = []
    train_txt = []
    train_input_length = []
    train_label_length = []
    train_orig_txt = []

    #lists for test dataset
    test_img = []
    test_txt = []
    test_input_length = []
    test_label_length = []
    test_orig_txt = []

    #lists for validation dataset
    valid_img = []
    valid_txt = []
    valid_input_length = []
    valid_label_length = []
    valid_orig_txt = []
    for i in range(len(data)):
        if i < 3926:
            valid_orig_txt.append(label[i])   
            valid_label_length.append(len(label[i]))
            valid_input_length.append(88)
            valid_img.append(data[i])
            valid_txt.append(encoded_label[i])
            valid_padded_txt.append(pad_label[i])

        elif i > 3926 and i < 7852:
            test_orig_txt.append(label[i])   
            test_label_length.append(len(label[i]))
            test_input_length.append(88)
            test_img.append(data[i])
            test_txt.append(encoded_label[i])
            test_padded_txt.append(pad_label[i])

        else:
            train_orig_txt.append(label[i])   
            train_label_length.append(len(label[i]))
            train_input_length.append(88)
            train_img.append(data[i])
            train_txt.append(encoded_label[i])
            train_padded_txt.append(pad_label[i])

    # creating model
    model = create_model(maxv)

    filepath="models/best_model.hdf5"
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    tb = TensorBoard(histogram_freq=1,write_grads=True)
    callbacks_list = [checkpoint,tb,es]

    # preparing input data etc
    training_img = np.array(train_img)
    train_input_length = np.array(train_input_length)
    train_label_length = np.array(train_label_length)
    train_padded_txt = np.array(train_padded_txt)

    valid_img = np.array(valid_img)
    valid_input_length = np.array(valid_input_length)
    valid_label_length = np.array(valid_label_length)
    valid_padded_txt = np.array(valid_padded_txt)

    test_img = np.array(test_img)
    test_input_length = np.array(test_input_length)
    test_label_length = np.array(test_label_length)
    test_padded_txt = np.array(test_padded_txt)

    # model training
    batch_size = 256
    epochs = 100
    model.fit(x=[training_img, train_padded_txt, train_input_length, train_label_length], y=np.zeros(len(training_img)), batch_size=batch_size, epochs = epochs, validation_data = ([valid_img, valid_padded_txt, valid_input_length, valid_label_length], [np.zeros(len(valid_img))]), verbose = 1, callbacks = callbacks_list)