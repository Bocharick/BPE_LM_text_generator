import youtokentome as yttm
import tensorflow as tf
import os
import sys
import random


if __name__ == "__main__":
    print("Tensorflow version:", tf.__version__, file=sys.stderr)

    try:
        LENGTH = int(sys.argv[1])
    except:
        print(
            "ERROR: Wrong value. You need to set argv[1] value for \'LENGTH\' variable as integer 1+ (for example: 1000)")
        exit()

    try:
        CHOOSE_FROM_N_ARGMAXES = int(sys.argv[2])
    except:
        print(
            "ERROR: Wrong value. You need to set argv[2] value for \'CHOOSE_FROM_N_ARGMAXES\' variable as integer 1+ (for example: 10)")
        exit()


    yttm_bpe_model_filepath = "../data/yttm.bpe"
    tfmodel_filepath = "../data/tfmodel.h5"
    assert os.path.isfile(yttm_bpe_model_filepath)
    assert os.path.isfile(tfmodel_filepath)

    bpe_model = yttm.BPE(yttm_bpe_model_filepath)
    tfmodel = tf.keras.models.load_model(tfmodel_filepath)

    EOS_code = 3
    # check correct <EOS> code
    # print(bpe_model.decode([EOS_code]))

    cur_code = EOS_code
    predictions = []
    for i in range(LENGTH):
        prediction = tfmodel.predict([cur_code])[0]
        sorted_args = prediction.argsort()[::-1]
        choosed_code = random.choice(sorted_args[:CHOOSE_FROM_N_ARGMAXES])
        predictions.append(int(choosed_code))
        cur_code = predictions[-1]

    decoded_string = bpe_model.decode(predictions)[0]
    new_decoded_string = decoded_string.replace("<EOS> ", ".\n").replace("<EOS>", ".\n")
    print(new_decoded_string)
