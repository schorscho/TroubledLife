import sys
import os
import glob
import logging.config
from time import time
from datetime import datetime
from math import sqrt

import numpy as np

import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, classification_report, confusion_matrix

from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint
from keras.layers import Input, LSTM, RepeatVector, Dense, SimpleRNN, TimeDistributed, Concatenate, CuDNNGRU, GRU
from keras.models import Model, load_model
from keras.utils.vis_utils import plot_model

import data_preparation as sf


DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': { 
        'standard': { 
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': { 
        'default': { 
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': { 
        'root': { 
            'handlers': ['default'],
            'level': 'INFO'
        },
        'TROUBLED_LIFE': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        },
    } 
}


logging.config.dictConfig(DEFAULT_LOGGING)


logger = logging.getLogger('TROUBLED_LIFE')


class TL:
    PROJECT_ROOT_DIR = os.environ['TL_ROOT_DIR']
    INPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT_DIR, "output")

    TROUBLED_TRAIN_FILE = "troubled_life_policy_train_data"
    CLEAN_TRAIN_FILE = "pure_life_policy_train_data"
    TROUBLED_TEST_FILE = "troubled_life_policy_test_data"
    CLEAN_VAL_FILE = "pure_life_policy_test_data"

    INPUT_DIM = 2
    DP = 'DP01R00'
    
    LATENT_DIM = 15
    MV = 'MV03R00'

    TROUBLED_MSE_THRESHOLD = 1000
    
    BATCH_SIZE = 64
    OV = 'OV01R00'

    START_EP = 0
    END_EP = 400
    LOAD_MODEL = 'MV03R00_OV01R00_DP01R00_TR009_20180309-163927_SE000_EP398-00022.9634'
    SAVE_MODEL = 'TR009'
    

def build_keras_model(timesteps):
    inputs = Input(shape=(timesteps, TL.INPUT_DIM), name='Input_Sequence')
    print(inputs.shape)
    enc_output, enc_state_h_1 = \
        SimpleRNN(units=200, activation='relu', return_state=True, return_sequences=True, name='Encoder_RNN_1')(inputs)
    print(enc_output.shape, enc_state_h_1.shape)
    enc_output, enc_state_h_2 = \
        SimpleRNN(units=200, activation='relu', return_state=True, return_sequences=True, name='Encoder_RNN_2')(enc_output)
    print(enc_output.shape, enc_state_h_2.shape)
    enc_output, enc_state_h_3 = \
        SimpleRNN(units=200, activation='relu', return_state=True, name='Encoder_RNN_3')(enc_output)
    print(enc_output.shape, enc_state_h_2.shape)
    encoded = Concatenate(axis=1, name='Encoder_State_Fusion')([enc_state_h_1, enc_state_h_2, enc_state_h_3])
    print(encoded.shape)

    encoded = Dense(TL.LATENT_DIM, name='Latent')(encoded)
    print(encoded.shape)

    decoded = RepeatVector(timesteps, name='Latent_Sequence')(encoded)
    print(decoded.shape)
    decoded = SimpleRNN(units=200, activation='relu', return_sequences=True, name='Decoder_RNN_1')(decoded)
    print(decoded.shape)
    decoded = SimpleRNN(units=200, activation='relu', return_sequences=True, name='Decoder_RNN_2')(decoded)
    print(decoded.shape)
    decoded = SimpleRNN(units=200, activation='relu', return_sequences=True, name='Decoder_RNN_3')(decoded)
    print(decoded.shape)
    decoded = TimeDistributed(Dense(TL.INPUT_DIM, name='Output'), name='Output_Sequence')(decoded)
    print(decoded.shape)

    model = Model(inputs, decoded)

    return model


def lr_schedule(ep):
    lr = 0.001

    lr = lr / (ep // 10 + 1)

    logger.info('New learning rate: %01.10f', lr)
    
    return lr


def compile_keras_model(model):
    #adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00, clipnorm=1.0) #epsilon=None (doesn't work)

    model.compile(optimizer='adam', loss='mse', metrics=None)

    return model


def time_it(start, end):
    h, r = divmod(end - start, 3600)
    m, s = divmod(r, 60)
    
    return "{:0>2}:{:0>2}:{:06.3f}".format(int(h), int(m), s)


def load_data(file_name):
    path = os.path.join(TL.INPUT_DIR, file_name + '.csv')

    data = sf.load_life_policy_data(path)

    return data


def load_all_data(clean_train_set, clean_val_set, troubled_test_set):
    clean_train_data = None
    clean_val_data = None
    troubled_test_data = None
    
    if clean_train_set:
        logger.info("Loading initial training data ...")

        clean_train_data = load_data(file_name=TL.CLEAN_TRAIN_FILE)

        logger.info("Loading initial training data done.")

    if clean_val_set:
        logger.info("Loading prepared validation data ...")

        clean_val_data = load_data(file_name=TL.CLEAN_VAL_FILE)

        logger.info("Loading prepared validation data done.")

    if troubled_test_set:
        logger.info("Loading prepared test data ...")

        troubled_test_data = load_data(TL.TROUBLED_TEST_FILE)

        logger.info("Loading prepared test data done.")

    return clean_train_data, clean_val_data, troubled_test_data


def get_data_packages(data):
    policy_histories_length, max_policy_history_length = \
        sf.get_policy_history_lengths(policy_histories=data)

    y, x_seq, _ = \
        sf.prepare_labels_features_lengths(policy_histories=data,
                                           policy_histories_lengths=policy_histories_length,
                                           max_policy_history_length=max_policy_history_length,
                                           binary_classification=True)

    return x_seq, y, max_policy_history_length


def load_keras_model(model_file):
    model = load_model(os.path.join(TL.OUTPUT_DIR, model_file + '.hdf5'))

    return model
    

def save_keras_model(model, model_file):
    model.save(os.path.join(TL.OUTPUT_DIR, model_file))


def save_history(history, date_string, save_model_as, start_epoch):
    file = '{0}_{1}_{2}'.format(TL.MV, TL.OV, TL.DP)
    file += '_' + save_model_as + '_' + date_string
    file += '_SE{0:03d}_history'.format(start_epoch)

    hist = pd.DataFrame.from_dict(history.history)
    hist['epoch'] = [i + 1 for i in range(len(hist))]
    hist.set_index('epoch', inplace=True)
    hist.to_csv(path_or_buf=os.path.join(TL.OUTPUT_DIR, file + '.csv'))

    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.yscale('log')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')

    fig = plt.gcf()
    fig.set_size_inches(12, 8)
    fig.savefig(os.path.join(TL.OUTPUT_DIR, file + '.png'), dpi=100)


def save_model_graph_and_summary(model):
    plot_model(model, to_file=os.path.join(TL.OUTPUT_DIR, '{0}.png'.format(TL.MV)), show_shapes=True)

    with open(os.path.join(TL.OUTPUT_DIR, '{0}.txt'.format(TL.MV)), 'w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))


def create_model_checkpoint_callbacks(save_model_as, date_string, start_epoch):
    file = '{0}_{1}_{2}'.format(TL.MV, TL.OV, TL.DP)
    file += '_' + save_model_as + '_' + date_string
    file += '_SE{0:03d}'.format(start_epoch)
    file += '_EP{epoch:03d}-{val_loss:010.4f}.hdf5'

    mc_callback = ModelCheckpoint(filepath=os.path.join(TL.OUTPUT_DIR, file),
                                  monitor='val_loss', verbose=0, save_best_only=True,
                                  save_weights_only=False, mode='min', period=1)

    fn_callback = Model_File_Name_Tracker(save_model_as, date_string, start_epoch)

    return mc_callback, fn_callback


class Model_File_Name_Tracker(Callback):
    def __init__(self, save_model_as, date_string, start_epoch):
        super(Callback, self).__init__()

        self.save_model_as = save_model_as
        self.date_string = date_string
        self.start_epoch = start_epoch
        self.best_val_loss = None
        self.best_file_name = None

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']

        if self.best_val_loss is None or self.best_val_loss > val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch

    def on_train_end(self, logs=None):
        self.best_file_name = '{0}_{1}_{2}'.format(TL.MV, TL.OV, TL.DP)
        self.best_file_name += '_' + self.save_model_as + '_' + self.date_string
        self.best_file_name += '_SE{0:03d}'.format(self.start_epoch)
        self.best_file_name += '_EP{0:03d}-{1:010.4f}'.format(self.best_epoch + 1, self.best_val_loss)


class Evaluation(Callback):
    def __init__(self, val_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.x_v, self.y_v = val_data

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.interval == 0:
            y_p = self.model.predict(self.x_v, verbose=0)

            y_p = np.reshape(y_p, (y_p.shape[0] * y_p.shape[1], y_p.shape[2]))
            y = np.reshape(self.y_v, (self.y_v.shape[0] * self.y_v.shape[1], self.y_v.shape[2]))

            skl_mse = mean_squared_error(y, y_p)
            skl_rmse = sqrt(skl_mse)

            print(" - val_skl_mse ({:.6f}), val_skl_rmse ({:.6f})".format(skl_mse, skl_rmse))


def execute_training(start_epoch, end_epoch, build_on_model, save_model_as, clean_train_data, clean_val_data):
    x_t, y_t, max_len_t = get_data_packages(clean_train_data)
    x_v, y_v, max_len_v = get_data_packages(clean_val_data)

    date_string = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    logger.info('Building/compiling model ...')

    if build_on_model is None:
        model = build_keras_model(timesteps=max(max_len_t, max_len_v))
    else:
        model = build_on_model

    model = compile_keras_model(model)

    callbacks = [LearningRateScheduler(lr_schedule), Evaluation(val_data=(x_v, x_v), interval=1)]

    fn_callback = None
    best_file_name = None

    if save_model_as is not None:
        mc_callback, fn_callback = create_model_checkpoint_callbacks(save_model_as, date_string, start_epoch)

        callbacks.append(mc_callback)
        callbacks.append(fn_callback)

        save_model_graph_and_summary(model)

    logger.info('Building/compiling model done.')

    logger.info('Fitting model ...')

    history = model.fit(
        x=[x_t], y=x_t,
        batch_size=TL.BATCH_SIZE,
        epochs=end_epoch,
        verbose=1,
        callbacks=callbacks,
        shuffle=True,
        initial_epoch=start_epoch,
        steps_per_epoch=None,
        validation_data=[[x_v], x_v])

    if save_model_as is not None:
        save_history(history, date_string, save_model_as, start_epoch)

        best_file_name = fn_callback.best_file_name

    logger.info('Fitting model done.')

    return model, best_file_name


def execute_test(model, troubled_test_data, model_file):
    logger.info("Testing model ...")

    x_t, y_t, _ = get_data_packages(troubled_test_data)

    x_p = model.predict(x_t, verbose=1)

    pol_mse = np.mean(np.mean(((x_t - x_p) ** 2), axis=2), axis=1)
    y_t = (y_t > 0)
    y_p = (pol_mse > TL.TROUBLED_MSE_THRESHOLD)
    certainty = abs(TL.TROUBLED_MSE_THRESHOLD - pol_mse) / TL.TROUBLED_MSE_THRESHOLD
    pred_correct = (y_p == y_t)

    test_result = pd.DataFrame(
        {'id': troubled_test_data.index.levels[0], 'label': y_t, 'pol_mse': pol_mse, 'label_pred': y_p,
         'pred_correct': pred_correct, 'certainty': certainty})
    test_result.set_index('id', inplace=True)
    test_result.to_csv(path_or_buf=os.path.join(TL.OUTPUT_DIR, model_file + '_results.csv'))

    print('\n')
    print('Accuracy: ', accuracy_score(y_t, y_p))
    print('Precision: ', precision_score(y_t, y_p))
    print('Recall: ', recall_score(y_t, y_p))
    print('\n')
    print(classification_report(y_t, y_p, target_names=['False', 'True']))
    print(confusion_matrix(y_t, y_p))
    print('\n')
    print(test_result[test_result["pred_correct"] == False])
    print('\n')


def main():
    overall = time()

    logger.info("Main script started ...")     
    
    training = False
    test = False

    model = None
    best_model_file = None
    
    for arg in sys.argv[1:]:
        if arg == 'training':
            training = True
        elif arg == 'test':
            test = True

    if not training and not test:
        training = True
    
    then = time()

    logger.info("Data preparation started ...")     

    clean_train_data, clean_val_data, troubled_test_data = load_all_data(
        clean_train_set=training,
        clean_val_set=training,
        troubled_test_set=test)

    logger.info("Data preparation done in %s.", time_it(then, time()))

    if (training or test) and TL.LOAD_MODEL is not None:
        logger.info("Loading model ...")

        model = load_keras_model(TL.LOAD_MODEL)

        logger.info("Loading model done.")

    if training:
        logger.info("Executing training ...")    

        then = time()
        
        logger.info("Begin training ...")

        model, best_model_file = execute_training(start_epoch=TL.START_EP, end_epoch=TL.END_EP,
                                 build_on_model=model, save_model_as=TL.SAVE_MODEL,
                                 clean_train_data=clean_train_data, clean_val_data=clean_val_data)

        logger.info("Training completed.")    
        
        logger.info("Done executing training in %s.", time_it(then, time()))    

    if test:
        logger.info("Executing test ...")    

        then = time()

        if best_model_file is not None:
            model = load_keras_model(best_model_file)
            model_file = best_model_file
        else:
            model_file = TL.LOAD_MODEL

        execute_test(model, troubled_test_data, model_file)
    
        logger.info("Done executing test in %s.", time_it(then, time()))    
        

    logger.info("Main script finished in %s.", time_it(overall, time()))
        

if __name__ == "__main__":
    main()
