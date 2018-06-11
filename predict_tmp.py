from model_tmp import *
from model_utils import *
from data_utils import _2uni, _2utf8, _2gbk

import argparse
parser = argparse.ArgumentParser(
    description="Predict with a saved model in the specified folder.")

parser.add_argument(
    "-l",
    dest="load_folder",
    type=str,
    help="The specified folder to load saved model. If not specified, the model will be initialized.")
parser.add_argument(
    "-f",
    dest="input_file",
    type=str,
    help="The specified input file to make predictions.")
parser.add_argument(
    "-n",
    dest="model_index",
    type=int,
    default=-1,
    help=".")
args = parser.parse_args()

if not os.path.isdir(args.load_folder):
    raise Exception("You should use --l to add the saved model's folder path. (or maybe you gave a wrong path)")
if not os.path.isfile(args.input_file):
    raise Exception("You should use --f to add the input file path. (or maybe you gave a wrong path)")

CONFIG = loadConfigFromFolder(None, args.load_folder)
full_dict_src, rev_dict_src = loadDict(CONFIG['SRC_DICT'])
full_dict_dst, rev_dict_dst = loadDict(CONFIG['DST_DICT'])

f_x = open(args.input_file,'r')
x_test_raw = f_x.readlines()
test_raw = [ [x_test_raw[i].strip(),''] for i in range(len(x_test_raw))]
test_buckets_raw = arrangeBuckets(test_raw, CONFIG['BUCKETS'])
print([len(b) for b in test_buckets_raw])
f_x.close()


with tf.Session() as sess:
    print('Loading model...')
    print('Result in Training: %.6f'%(CONFIG['LOG'][args.model_index]))
    CONFIG['IS_TRAIN'] = False
    CONFIG['INPUT_DROPOUT'] = 1.0
    CONFIG['OUTPUT_DROPOUT'] = 1.0
    Model = instanceOfInitModel(sess, CONFIG)
    loadModelFromFolder(sess, Model.saver, CONFIG, args.load_folder, args.model_index)

    test_results=dict()
    for b in range(len(CONFIG['BUCKETS'])):
        n_b = len(test_buckets_raw[b])
        for k in range((n_b+CONFIG['BATCH_SIZE']-1)/CONFIG['BATCH_SIZE']):
            test_batch = [ test_buckets_raw[b][i%n_b] for i in range(k*CONFIG['BATCH_SIZE'], (k+1)*CONFIG['BATCH_SIZE']) ]
            print('test process: [%d/%d] [%d/%d]'%(b+1, len(CONFIG['BUCKETS']), k*CONFIG['BATCH_SIZE'], n_b))
            test_batch = map(list, zip(*test_batch))
            model_inputs, len_inputs, inputs_mask = dataSeqs2NpSeqs(test_batch[0], full_dict_src, CONFIG['BUCKETS'][b][0])
            model_outputs, len_outputs, outputs_mask = dataSeqs2NpSeqs(test_batch[1], full_dict_dst, CONFIG['BUCKETS'][b][1])
            model_targets, len_targets, targets_mask = dataSeqs2NpSeqs(test_batch[1], full_dict_dst, CONFIG['BUCKETS'][b][1], bias=1)
            predict_outputs = Model.test_on_batch(sess, model_inputs, len_inputs, inputs_mask, model_outputs, len_outputs, outputs_mask, model_targets, len_targets, targets_mask)

            test_batch = map(list, zip(*test_batch))
            for i in range(CONFIG['BATCH_SIZE']):
                test_results[test_batch[i][0]] = dataLogits2Seq(predict_outputs[i], rev_dict_dst, calc_argmax=False)
                if random.random()<0.01:
                    try:
                        print('Raw input: %s\nExpected output: %s\nModel output: %s' % (test_batch[i][0], test_batch[i][1], test_results[test_batch[i][0]]))
                    except UnicodeDecodeError:
                        pass
    f_x = open(args.input_file,'r')
    fname = '/predictions_%d_%.2f.txt'%(args.model_index, CONFIG['LOG'][args.model_index])
    f_y = open(args.load_folder+fname,'w')
    for line in f_x.readlines():
        s = test_results[line.strip()]
        # s = s.replace('<UNK> ', '')
        p = s.find('<EOS>')
        if p==-1:
            p = len(s)
        f_y.write(s[:p]+'\n')
    f_x.close()
    f_y.close()
    print('Prediction completed! Please check the results @ %s'%(args.load_folder+fname))
