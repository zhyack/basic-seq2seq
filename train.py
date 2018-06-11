from model_tmp import *
from model_utils import *
from data_utils import _2uni, _2utf8, _2gbk

CONFIG = dict()

CONFIG['LR'] = 1.0
CONFIG['WE_LR'] = 0.00001
CONFIG['ENCODER_LR'] = 0.00001
CONFIG['DECODER_LR'] = 0.00001
CONFIG['SPLIT_LR'] = False
CONFIG['LR_DECAY'] =  0.7 #0.98
CONFIG['OPTIMIZER'] = 'GD'
CONFIG['CELL'] = "lstm"
CONFIG['WORD_EMBEDDING_SIZE'] = 500
CONFIG['ENCODER_HIDDEN_SIZE'] = 500
CONFIG['DECODER_HIDDEN_SIZE'] = 500
CONFIG['ENCODER_LAYERS'] = 1
CONFIG['DECODER_LAYERS'] = 2
CONFIG['BIDIRECTIONAL_ENCODER'] = True
CONFIG['ATTENTION_DECODER'] = True
# CONFIG['ATTENTION_MECHANISE'] = 'BAHDANAU'
CONFIG['ATTENTION_MECHANISE'] = 'LUONG'
CONFIG['INPUT_DROPOUT'] = 1.0
CONFIG['OUTPUT_DROPOUT'] = 0.7
CONFIG['CLIP']=True
CONFIG['MAX_STEPS_PER_ITER']=6000
CONFIG['DECAY_STEPS']=480000
CONFIG['RL_ENABLE']=False
CONFIG['BLEU_RL_ENABLE']=False
CONFIG['RL_RATIO']=0.4

CONFIG['CLIP_NORM']=1.0
CONFIG['VAR_NORM_BETA']=0.00003
CONFIG['TRAIN_ON_EACH_STEP']=True
CONFIG['ITERS']=500
CONFIG['BATCH_SIZE']=64

CONFIG['SEED'] = 233333

CONFIG['SRC_DICT']='data/wmt15-de-en/src.dict'
CONFIG['DST_DICT']='data/wmt15-de-en/tgt.dict'
CONFIG['TRAIN_INPUT']='data/wmt15-de-en/wmt15-all-de-en.en.tok'
CONFIG['TRAIN_OUTPUT']='data/wmt15-de-en/wmt15-all-de-en.de.tok'
CONFIG['DEV_INPUT']='data/wmt15-de-en/newstest2013.en.tok'
CONFIG['DEV_OUTPUT']='data/wmt15-de-en/newstest2013.de.tok'
CONFIG['HYP_FILE_PATH']='data/'
CONFIG['REF_FILE_PATH_FORMAT']='data/'


CONFIG['GLOBAL_STEP']=1
CONFIG['MAX_IN_LEN']=55
CONFIG['MAX_OUT_LEN']=55
# CONFIG['BUCKETS']=[[10,15],[20,30],[30,40],[40,50],[50,60],[60,60]]
CONFIG['BUCKETS']=[[60,60]]

CONFIG['USE_BS']=True
CONFIG['BEAM_WIDTH']=5

CONFIG['LOG']=[]


import argparse
parser = argparse.ArgumentParser(
    description="Train a seq2seq model and save in the specified folder.")
parser.add_argument(
    "-s",
    dest="save_folder",
    type=str,
    default=None,
    help="The specified folder to save. If not specified, the model will not be saved.")
parser.add_argument(
    "-l",
    dest="load_folder",
    type=str,
    help="The specified folder to load saved model. If not specified, the model will be initialized.")
args = parser.parse_args()



full_dict_src, rev_dict_src = loadDict(CONFIG['SRC_DICT'])
full_dict_dst, rev_dict_dst = loadDict(CONFIG['DST_DICT'])
CONFIG['INPUT_VOCAB_SIZE']=len(rev_dict_src)
CONFIG['OUTPUT_VOCAB_SIZE']=len(rev_dict_dst)
print(CONFIG['INPUT_VOCAB_SIZE'],CONFIG['OUTPUT_VOCAB_SIZE'])
CONFIG['ID_END']=full_dict_dst['<EOS>']
CONFIG['ID_BOS']=full_dict_dst['<BOS>']
CONFIG['ID_PAD']=full_dict_dst['<PAD>']
CONFIG['ID_UNK']=full_dict_dst['<UNK>']


f_x = open(CONFIG['TRAIN_INPUT'],'r')
x_train_raw = f_x.readlines()
f_y = open(CONFIG['TRAIN_OUTPUT'],'r')
y_train_raw = f_y.readlines()
train_raw = [ [x_train_raw[i].strip(),y_train_raw[i].strip()] for i in range(len(x_train_raw))]
train_buckets_raw = arrangeBuckets(train_raw, CONFIG['BUCKETS'])
print([len(b) for b in train_buckets_raw])
f_x.close()
f_y.close()
f_x = open(CONFIG['DEV_INPUT'],'r')
x_eval_raw = f_x.readlines()
f_y = open(CONFIG['DEV_OUTPUT'],'r')
y_eval_raw = f_y.readlines()
eval_raw = [ [x_eval_raw[i].strip(),y_eval_raw[i].strip()] for i in range(len(x_eval_raw))]
# eval_raw = eval_raw[:256]
eval_buckets_raw = arrangeBuckets(eval_raw, CONFIG['BUCKETS'])
print([len(b) for b in eval_buckets_raw])
f_x.close()
f_y.close()


g = tf.Graph()
with g.as_default():
    tf.set_random_seed(CONFIG['SEED'])
    random.seed(CONFIG['SEED'])
    with tf.Session() as sess:
        print('Loading model...')
        if args.load_folder != None:
            old_CONFIG = loadConfigFromFolder(None, args.load_folder)
            if old_CONFIG != None:
                CONFIG = old_CONFIG
        CONFIG['IS_TRAIN'] = True
        Model = instanceOfInitModel(sess, CONFIG)
        if args.load_folder != None:
            CONFIG = loadModelFromFolder(sess, Model.saver, CONFIG, args.load_folder, -1)
        tf.set_random_seed(CONFIG['SEED'])
        random.seed(CONFIG['SEED'])
        print('Training Begin...')
        log_losses = CONFIG['LOG']
        print(log_losses)
        for n_iter in range(CONFIG['GLOBAL_STEP']/CONFIG['MAX_STEPS_PER_ITER'], CONFIG['ITERS']):
            while True:
                b = random.randint(0, min(len(CONFIG['BUCKETS'])-1, n_iter))
                n_b = len(train_buckets_raw[b])
                train_batch = [ train_buckets_raw[b][random.randint(0, n_b-1)] for _ in range(CONFIG['BATCH_SIZE'])]
                train_batch = map(list, zip(*train_batch))
                model_inputs, len_inputs, inputs_mask = dataSeqs2NpSeqs(train_batch[0], full_dict_src, CONFIG['BUCKETS'][b][0])
                model_outputs, len_outputs, outputs_mask = dataSeqs2NpSeqs(train_batch[1], full_dict_dst, CONFIG['BUCKETS'][b][1])
                model_targets, len_targets, targets_mask = dataSeqs2NpSeqs(train_batch[1], full_dict_dst, CONFIG['BUCKETS'][b][1], bias=1)
                batch_loss = Model.train_on_batch(sess, model_inputs, len_inputs, inputs_mask, model_outputs, len_outputs, outputs_mask, model_targets, len_targets, targets_mask)
                if CONFIG['RL_ENABLE']:
                    if CONFIG['BLEU_RL_ENABLE']:
                        print('Train completed for Iter@%d, Step@%d: CE_Loss=%.6f RL_Loss=%.6f BLEU_Loss=%.6f Loss=%.6f LR=%.8f'%(n_iter, CONFIG['GLOBAL_STEP'], batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[0]+batch_loss[1]+batch_loss[2], CONFIG['LR']))
                    else:
                        print('Train completed for Iter@%d, Step@%d: CE_Loss=%.6f RL_Loss=%.6f Loss=%.6f LR=%.8f'%(n_iter, CONFIG['GLOBAL_STEP'], batch_loss[0], batch_loss[1], batch_loss[0]+batch_loss[1], CONFIG['LR']))
                else:
                    print('Train completed for Iter@%d, Step@%d: Loss=%.6f LR=%.8f'%(n_iter, CONFIG['GLOBAL_STEP'], batch_loss, CONFIG['LR']))
                CONFIG['GLOBAL_STEP']+=1
                if (CONFIG['GLOBAL_STEP'] % CONFIG['MAX_STEPS_PER_ITER'] == 0):
                    break
            if CONFIG['GLOBAL_STEP']%CONFIG['DECAY_STEPS']==0:
                CONFIG['LR'] = CONFIG['LR']*CONFIG['LR_DECAY']
                sess.run(Model.lr_decay_op)
                CONFIG['DECAY_STEPS']=60000
            print('Iter@%d completed! Start Evaluating...'%(n_iter))
            eval_losses=[]
            eval_results=dict()
            # eval_buckets_raw = train_buckets_raw
            for b in range(len(CONFIG['BUCKETS'])):
                n_b = len(eval_buckets_raw[b])
                for k in range((n_b+CONFIG['BATCH_SIZE']-1)/CONFIG['BATCH_SIZE']):
                    eval_batch = [ eval_buckets_raw[b][i%n_b] for i in range(k*CONFIG['BATCH_SIZE'], (k+1)*CONFIG['BATCH_SIZE']) ]
                    print('Eval process: [%d/%d] [%d/%d]'%(b+1, len(CONFIG['BUCKETS']), k*CONFIG['BATCH_SIZE'], n_b))
                    eval_batch = map(list, zip(*eval_batch))
                    model_inputs, len_inputs, inputs_mask = dataSeqs2NpSeqs(eval_batch[0], full_dict_src, CONFIG['BUCKETS'][b][0])
                    model_outputs, len_outputs, outputs_mask = dataSeqs2NpSeqs(eval_batch[1], full_dict_dst, CONFIG['BUCKETS'][b][1])
                    model_targets, len_targets, targets_mask = dataSeqs2NpSeqs(eval_batch[1], full_dict_dst, CONFIG['BUCKETS'][b][1], bias=1)
                    batch_loss, predict_outputs = Model.eval_on_batch(sess, model_inputs, len_inputs, inputs_mask, model_outputs, len_outputs, outputs_mask, model_targets, len_targets, targets_mask)

                    eval_losses.append(batch_loss)
                    eval_batch = map(list, zip(*eval_batch))
                    for i in range(CONFIG['BATCH_SIZE']):
                        eval_results[eval_batch[i][0]] = dataLogits2Seq(predict_outputs[i], rev_dict_dst, calc_argmax=False)
                        if random.random()<0.01:
                            try:
                                print('Raw input: %s\nExpected output: %s\nModel output: %s' % (eval_batch[i][0], eval_batch[i][1], eval_results[eval_batch[i][0]]))
                                # for j in range(len(predict_outputs[i])):
                                #     ll = eval_batch[i][1].split()
                                #     try:
                                #         print('<PAD:%.6f>, <%s:%.6f>' % (predict_outputs[i][j][full_dict_dst['<PAD>']], 'true', predict_outputs[i][j][full_dict_dst[ll[j]]]))
                                #     except:
                                #         pass
                            except UnicodeDecodeError:
                                pass


            f_x = open(CONFIG['DEV_INPUT'],'r')
            while(True):
                f_lock = open('tmp/lock','r')
                l = f_lock.readline().strip()
                f_lock.close()
                if l != 'LOCKED':
                    f_lock = open('tmp/lock','w')
                    f_lock.write('LOCKED')
                    f_lock.close()
                    break

            f_y = open('tmp/predictions.txt','w')
            for line in f_x.readlines():
                s = eval_results[line.strip()]
                p = s.find('<EOS>')
                if p==-1:
                    p = len(s)
                f_y.write(s[:p]+'\n')
            f_x.close()
            f_y.close()
            eval_bleu=0.0
            # eval_bleu = bleuPerlInstance()
            eval_bleu = bleuPerlInstance2()
            f_lock = open('tmp/lock','w')
            f_lock.close()
            print eval_bleu

            print('Evaluation completed:\nAverage Loss:%.6f\nBLEU:%.2f'%(sum(eval_losses)/len(eval_losses),eval_bleu))
            print(log_losses[max(0,len(log_losses)-200):])
            log_losses.append(sum(eval_losses)/len(eval_losses))
            if log_losses[-1]<=min(log_losses[max(0,len(log_losses)-3):]):
                # sess.run(Model.lr_decay_op)
                print('Learning rate turn down-to %.6f'%(sess.run(Model.learning_rate)))
            if args.save_folder != None:
                CONFIG['LOG']=log_losses
                saveModelToFolder(sess, Model.saver, args.save_folder, CONFIG, n_iter)
        print('Training Completed...')
