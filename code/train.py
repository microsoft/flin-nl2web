import math
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from code.nsm_model.util_helper import *


def train_and_evaluate(sess, args, model, train, valid, saver, model_path, batch_size, vocab, embed_dim, pre_train = False):

    print('starting training ...')
    train_act = train[0]
    train_para_tag = train[1]
    train_para_match = train[2]

    valid_act = valid[0]
    valid_para_tag = valid[1]
    valid_para_match = valid[2]

    print('train_size_para: ', len(train_para_tag))
    print('valid_size_para: ', len(valid_para_tag))

    if bool(args['train_mode']):
        num_batches_act = int(math.ceil((len(train_act) * 1.0)/batch_size))
        p2_batch_size = batch_size
        num_batches_para_tag = int(math.ceil((len(train_para_tag) * 1.0) / p2_batch_size))
        num_batches_para_match = int(math.ceil((len(train_para_match) * 1.0) / p2_batch_size))
        print('num_batches ..', num_batches_act, num_batches_para_tag, num_batches_para_match)

        # start training ...
        for i in range(int(args['max_epoch'])):
            # shuffle train dataset ...
            train_shuff_act = get_shuffled_train_data(train_act)
            train_shuff_para_tag = get_shuffled_train_data(train_para_tag)
            train_shuff_para_match = get_shuffled_train_data(train_para_match)

            #print(i)
            if i % 3 == 2:
                #print(i)
                '''=============================================== TRAIN P1 ================================================'''
                # mini-batch training ...
                avg_loss = 0
                avg_post_dist = 0.0
                avg_neg_dist = 0.0
                avg_acc = 0.0
                for j in range(num_batches_act):
                    batch_X_action = get_next_batch_action(train_shuff_act, j, batch_size)

                    # print(batch_X_pos_action_para.shape)
                    # print(batch_X_pos_action_para[0])

                    _, loss, pos_dist, neg_dist, acc = model.train_action(batch_X_action)
                    avg_loss += loss
                    avg_post_dist += pos_dist
                    avg_neg_dist += neg_dist
                    avg_acc += acc
                    sys.stderr.write("\r")
                    sys.stderr.write("P1_Act_Epoch- %d processing %0.3f" % ((i+1), round(j*100.0/num_batches_act, 3)))
                    sys.stderr.flush()

                # mini-batch evaluation ...
                vd_batch_size = batch_size
                num_vd_batches_act = int(math.ceil(len(valid_act) / vd_batch_size))

                avg_vd_loss = 0
                avg_vd_pos = 0.0
                avg_vd_neg = 0.0
                avg_vd_acc = 0.0
                for k in range(num_vd_batches_act):
                    batch_X_action_vd = get_next_batch_action(valid_act, k, vd_batch_size)

                    # print(batch_vd_X_pos_action_para.shape)
                    # print(batch_vd_X_pos_action_para[0])

                    valid_loss, vd_pos, vd_neg, vd_acc = model.evaluate_action(batch_X_action_vd)

                    avg_vd_loss += valid_loss
                    avg_vd_pos += vd_pos
                    avg_vd_neg += vd_neg
                    avg_vd_acc += vd_acc

                print('\ttr_loss: ', round(avg_loss, 3), ' tr_acc: ', round((avg_acc/num_batches_act), 3),
                      ' tr_pos: ', round((avg_post_dist/num_batches_act), 3), ' tr_neg: ', round((avg_neg_dist/num_batches_act), 3),
                      ' ------ vd_loss: ', round(avg_vd_loss, 3), ' vd_acc: ', round((avg_vd_acc/num_vd_batches_act), 3),
                      ' vd_pos: ', round((avg_vd_pos/num_vd_batches_act), 3), ' vd_neg: ', round((avg_vd_neg/num_vd_batches_act), 3))

            ''' =============================================== TRAIN P2 ================================================ '''

            if i % 8 == 0:
                # mini-batch training ...
                avg_loss_para_tag = 0
                avg_acc_para_tag = 0.0

                for j in range(num_batches_para_tag):
                    batch_X_para_tag = get_next_batch_para_tagger(train_shuff_para_tag, j, p2_batch_size)

                    # for dict_key in batch_X_para_tag:
                    #     if not dict_key.startswith('bert'):
                    #         print(dict_key, batch_X_para_tag[dict_key].shape)

                    _, loss_para_tag, acc_para_tag \
                        = model.train_para_tag(batch_X_para_tag)

                    # acc_para_tag = compute_taging_accuracy(batch_X_para_tag['tag_label'],
                    #                                                batch_X_para_tag['q_len'], viterbi_sequence)

                    avg_loss_para_tag += loss_para_tag
                    avg_acc_para_tag += acc_para_tag

                    sys.stderr.write("\r")
                    sys.stderr.write("P2_Tag_Epoch- %d processing %0.3f" % ((i + 1), round(j * 100.0 / num_batches_para_tag, 3)))
                    sys.stderr.flush()

                # mini-batch evaluation ...
                vd_batch_size = batch_size * 2
                num_vd_batches_para_tag = int(math.ceil((len(valid_para_tag) * 1.0) / vd_batch_size))

                avg_vd_loss_para_tag = 0.0
                avg_vd_acc_para_tag = 0.0

                for k in range(num_vd_batches_para_tag):
                    batch_X_para_vd_tag = get_next_batch_para_tagger(valid_para_tag, k, vd_batch_size)

                    vd_loss_para_tag, vd_accuracy_para_tag, \
                    sequence_output = model.evaluate_para_tag(batch_X_para_vd_tag)

                    # vd_accuracy_para_tag = compute_taging_accuracy(batch_X_para_vd_tag['tag_label'],
                    #                         batch_X_para_vd_tag['q_len'], vd_viterbi_sequence)

                    avg_vd_loss_para_tag += vd_loss_para_tag
                    avg_vd_acc_para_tag += vd_accuracy_para_tag

                print('\ttr_loss: ', round(avg_loss_para_tag, 3),
                      ' tr_acc: ', round((avg_acc_para_tag / num_batches_para_tag), 3),
                      ' ------ vd_loss: ', round(avg_vd_loss_para_tag, 3), ' vd_acc: ',
                      round((avg_vd_acc_para_tag / num_vd_batches_para_tag), 3))

            # ===============================================================================================

            if i % 1 == 0:
                # mini-batch training ...
                avg_loss_para = 0
                avg_pos_dist_para = 0.0
                avg_neg_dist_para = 0.0
                avg_acc_para = 0.0

                for j in range(num_batches_para_match):
                    batch_X_para = get_next_batch_para_matcher(train_shuff_para_match, j, p2_batch_size)

                    _, loss_para, pos_dist_para, neg_dist_para, accuracy_para \
                                = model.train_para_match(batch_X_para)

                    avg_loss_para += loss_para
                    avg_pos_dist_para += pos_dist_para
                    avg_neg_dist_para += neg_dist_para
                    avg_acc_para += accuracy_para

                    sys.stderr.write("\r")
                    sys.stderr.write("P2_Mat_Epoch- %d processing %0.3f" % ((i + 1),
                                                                            round(j * 100.0 / num_batches_para_match, 3)))
                    sys.stderr.flush()

                # mini-batch evaluation ...
                vd_batch_size = batch_size
                num_vd_batches_para_match = int(math.ceil(len(valid_para_match) / vd_batch_size))

                avg_vd_loss_para_match = 0
                avg_vd_acc_para_match = 0.0
                avg_vd_pos_dist_para = 0.0
                avg_vd_neg_dist_para = 0.0

                for k in range(num_vd_batches_para_match):
                    batch_X_para_vd_match = get_next_batch_para_matcher(valid_para_match, k, vd_batch_size)

                    vd_loss_para, vd_pos_dist_para, vd_neg_dist_para,\
                    vd_accuracy_para = model.evaluate_para_match(batch_X_para_vd_match)

                    avg_vd_loss_para_match += vd_loss_para
                    avg_vd_acc_para_match += vd_accuracy_para
                    avg_vd_pos_dist_para += vd_pos_dist_para
                    avg_vd_neg_dist_para += vd_neg_dist_para

                print('\ttr_loss: ', round(avg_loss_para, 3), ' tr_acc: ', round((avg_acc_para / num_batches_para_match), 3),
                      ' tr_pos: ', round((avg_pos_dist_para / num_batches_para_match), 3),
                      ' tr_neg: ', round((avg_neg_dist_para / num_batches_para_match), 3),
                      ' ------ vd_loss: ', round(avg_vd_loss_para_match, 3),
                      ' vd_acc: ', round((avg_vd_acc_para_match / num_vd_batches_para_match), 3),
                      ' vd_pos: ', round((avg_vd_pos_dist_para / num_vd_batches_para_match), 3),
                      ' vd_neg: ', round((avg_vd_neg_dist_para / num_vd_batches_para_match), 3))

        print('.... Training Complete .....')
        print('saving model ..')
        train_trace_id = args['train_trace_id']
        save_model(sess, saver, str(train_trace_id) + '_model', model_path)

    print("Finish.")