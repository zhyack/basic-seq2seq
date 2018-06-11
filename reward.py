from data_utils import *
from data_utils import _2uni, _2utf8, _2gbk
import os

def runMayGetValue(command_s, outputParser = None):
    outputs = os.popen(command_s)
    if outputParser==None:
        try:
            return float(outputs[0].strip())
        except TypeError,ValueError:
            print outputs
            return None
    return outputParser(outputs)

def bleuPerlParser(sl):
    ret = None
    print sl
    for s in sl:
        if s.find('BLEU')!=-1 and s.find('BP')!=-1 and s.find('ratio')!=-1 and s.find('hyp_len')!=-1 and s.find('ref_len')!=-1:
            st = s.find('=')+1
            en = s.find(',', st)
            ret = float(s[st:en])
            break
    return ret


def bleuPerlInstance():
    s = ""
    for i in range(1):
        s += 'data/wmt15-de-en/dev_ref/newstest2014.de.ref_%d '%(i)
    command_s = 'python multi_bleu.py -hyp tmp/predictions.txt -ref %s'%(s)
    p = bleuPerlParser
    return runMayGetValue(command_s, p)

def bleuPerlInstance2():
    s = ""
    for i in range(1):
        s += 'data/wmt15-de-en/dev_ref/newstest2013.de.ref_%d '%(i)
    command_s = 'python multi_bleu.py -hyp tmp/predictions.txt -ref %s'%(s)
    p = bleuPerlParser
    return runMayGetValue(command_s, p)

global dict_src, rev_dict_src, dict_dst, rev_dict_dst
dict_src, rev_dict_src, dict_dst, rev_dict_dst = None, None, None, None

def contentPenalty(inputs, outputs, SRC_DICT, DST_DICT, targets):
    # print(inputs.shape)
    # print(outputs.shape)
    global dict_src, rev_dict_src, dict_dst, rev_dict_dst
    if dict_dst==None or dict_dst==None or rev_dict_dst==None or rev_dict_src==None:
        dict_src, rev_dict_src = loadDict(SRC_DICT)
        dict_dst, rev_dict_dst = loadDict(DST_DICT)
    batch_size = len(inputs)
    assert(batch_size == len(outputs))
    max_len = outputs.shape[1]
    ret = []
    all_keys = [False]*len(dict_dst)
    for ind_src in rev_dict_src:
        word = rev_dict_src[ind_src]
        if word.upper()==word and len(word)>2 and word!='<EOS>' and dict_dst.has_key(word):
            ind_dst = dict_dst[word]
            all_keys[ind_dst]=True
    for i in range(batch_size):
        # sb = copy.deepcopy(score_board)
        expect_eos = np.argwhere(targets[i]==dict_dst['<EOS>'])[0]
        poskey_cnt = 0
        pos_keys = [0]*len(dict_dst)
        for ind in inputs[i]:
            ind_src = int(ind)
            word = rev_dict_src[ind_src]
            if word.upper()==word and len(word)>2 and word!='<PAD>' and word!='<EOS>' and dict_dst.has_key(word):
                ind_dst = dict_dst[word]
                if pos_keys[ind_dst]!=2:
                    poskey_cnt += 1
                pos_keys[ind_dst]=2
        ret.append([])
        predictions = outputs[i].argmax(axis=-1)
        eos = False
        for j in range(max_len):
            score_board = [0.0]*len(dict_dst)
            if eos:
                ret[i].append(score_board)
                continue
            p = int(predictions[j])
            if (all_keys[p] and pos_keys[p]==0):
                score_board[p] = 0.2
            elif (all_keys[p] and pos_keys[p]==2):
                score_board[p] = 2.0
                pos_keys[p] -= 1
            elif (all_keys[p] and pos_keys[p]==1):
                score_board[p] = 1.0
                pos_keys[p] -= 1
            else:
                score_board[p] = 1.0

            if j>0 and predictions[j]==predictions[j-1]:
                score_board[p] -= 1.0
            ret[i].append(score_board)
        remain_poskey_cnt = 0
        for k in range(len(dict_dst)):
            if pos_keys[k] == 2:
                remain_poskey_cnt += 1
        remain_penalty = remain_poskey_cnt*-0.5/poskey_cnt
        for j in range(max_len):
            p = int(predictions[j])
            if ret[i][j][p] != 2.0:
                ret[i][j][p] += remain_penalty
                ret[i][j][p] = max(0.0,ret[i][j][p])
    return np.array(ret, dtype=np.float32)

ref_dict = None
import bleu
import math
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))
def bleuPenalty(inputs, outputs, SRC_DICT, DST_DICT, HYP_FILE_PATH, REF_FILE_PATH_FORMAT):
    global dict_src, rev_dict_src, dict_dst, rev_dict_dst
    if dict_dst==None or dict_dst==None or rev_dict_dst==None or rev_dict_src==None:
        dict_src, rev_dict_src = loadDict(SRC_DICT)
        dict_dst, rev_dict_dst = loadDict(DST_DICT)
    global ref_dict
    if ref_dict == None:
        print('Loading references...')
        ref_dict = dict()
        hyp_list = []
        linecnt = 0
        f = open(HYP_FILE_PATH, 'r')
        for line in f.readlines():
            line = _2uni(line.strip())
            line = line.split()
            if len(line)>48:
                line = line[:48]
                line = ' '.join(line)
                line = line.replace('_',' ').replace(' ','')
                line = line[:150]
            else:
                line = ' '.join(line)
                line = line.replace('_',' ').replace(' ','')
                line = line[:150]
            ref_dict[line]=[]
            hyp_list.append(line)
            linecnt += 1
        f.close()
        ref_file_cnt = 0
        while(True):
            if os.path.isfile(REF_FILE_PATH_FORMAT%(ref_file_cnt)):
                f = open(REF_FILE_PATH_FORMAT%(ref_file_cnt), 'r')
                linecnt = 0
                for line in f.readlines():
                    line = _2uni(line.strip())
                    if len(line)>0:
                        ref_dict[hyp_list[linecnt]].append(line)
                    linecnt += 1
                ref_file_cnt+=1
            else:
                break

    batch_size = len(inputs)
    max_len = outputs.shape[1]
    ret = []
    for i in range(batch_size):
        ret.append([])
        predictions = outputs[i].argmax(axis=-1)
        last_bleu = 0.0
        src = ' '.join([_2uni(rev_dict_src[k]) for k in inputs[i]])
        src = src.replace('_',' ').replace(' ','')
        if src.find('<EOS>')!=-1:
            src = src[5:src.find('<EOS>')]
        else:
            src = src[5:]
        if len(src) > 150:
            src = src[:150]
        hyp = ' '.join([_2uni(rev_dict_dst[p]) for p in predictions])
        try:
            bleu_scores = bleu.incremental_sent_bleu(hyp,ref_dict[src])
        except KeyError:
            print(len(src))
        for j in range(max_len):
            bleu_score = bleu_scores[j]
            p = int(predictions[j])
            score_board = [0.0]*len(dict_dst)
            score_board[p] = sigmoid(last_bleu-bleu_scores[0])
            ret[i].append(score_board)
    return np.array(ret, dtype=np.float32)
