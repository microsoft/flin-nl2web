import _pickle as pickle
import pprint as pp, argparse

from code.evaluation import run_evaluation
from code.nsm_model.Neural_Semantic_Matcher import *
from code.nsm_model.util_helper import *
from code.train import train_and_evaluate
from code.train_data_preprocessing.prepare_vectorized_dataset import prepare_vectorized_dataset


def load_data(args):
    resource_dir = args['resource_dir']
    train_trace_id = args['train_trace_id']
    basepath = args['data_path'] + str(train_trace_id) + '_datafiles/'

    if os.path.isfile(resource_dir+str(train_trace_id)+'_data_vec_dump.pickle'):
        print('vectorized Data dump exists and loading...!')
        with open(resource_dir+str(train_trace_id)+'_data_vec_dump.pickle', "rb") as input_file:
            data_vec = pickle.load(input_file)
            print('... DATA Preparation Finished ....')
            return data_vec
    else:
        dataset_dump = None
        print('vectorized Data dump does not exist ...preparing vectorized training/evaluation data ...')
        if os.path.isfile(basepath + str(train_trace_id) + '_dataset_dump.pickle'):
            with open(basepath + str(train_trace_id) + '_dataset_dump.pickle', "rb") as input_file:
                dataset_dump = pickle.load(input_file)
                print('... DATA Preparation Finished ....')
        else:
            print("labeled data dump does not exit!")
            exit(0)

        data_vec = prepare_vectorized_dataset(train_trace_id, dataset_dump)
        return data_vec


def load_model(args, data_vec_dump):
    train_trace_id = args['train_trace_id']
    print('Loading model ...')

    # loading training dataset and vocab ....
    train, valid, vocab_to_id, id_to_vocab = data_vec_dump

    vocab_size = len(vocab_to_id)
    print('vocab_size: ', vocab_size)
    print('seq_len: ', max_seq_len)
    print('seq_len: ', max_no_dict)

    # run prediction model....
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
    with tf.device('/gpu:0'):
        # setup device
        pass

    # open tf session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                                  log_device_placement=True,
                                                                   gpu_options=gpu_options))

    # gpu_options=gpu_options))
    np.random.seed(int(args['random_seed']))
    tf.set_random_seed(int(args['random_seed']))
    model_path = args['model_dir']
    embed_dim = int(args['embed_dim'])
    vocab_size = len(vocab_to_id)

    model = None
    if args['encoder_type'] in ['rnn_uni', 'rnn_bi', 'cnn']:
       model = Neural_Semantic_Matching(sess, vocab_size, embed_dim, float(args['model_lr']), float(args['lamda']), args)
       print('model built ...')
    else:
       print('model type arg error')
       exit()

    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()
    model_exists = load_or_initialize_model(sess, saver, str(train_trace_id) + '_model', model_path)
    model.print_variable_names()

    if not model_exists:
        print('model does not exist!')

        if bool(args['train_mode']):
            print(' ....Learning Stated ....')
            batch_size = int(args['minibatch_size'])
            train_and_evaluate(sess, args, model, train, valid, saver, model_path, batch_size, vocab_to_id, embed_dim)
        else:
            exit(0)

    return sess, model


def main(args):
    data_dump = load_data(args)
    # model = None
    # sess = None
    if not data_dump:
        exit()
    sess, model = load_model(args, data_dump)

    # run evaluation ..
    run_evaluation(args, model, data_dump)

    # close the session
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for Graph Embedding QA agent')

    # agent parameters
    parser.add_argument('--model-lr', help='network learning rate', default=0.0001)
    parser.add_argument('--lamda', help='network regularization parameter', default=0.001)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=50)
    parser.add_argument('--max-epoch', help='No. of epoch', default=22)
    parser.add_argument('--embed-dim', help='Emdedding Dimension', default=300)

    # run parameters
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--model-dir', help='directory for storing learned QA_models',
                                                                                 default='./qa_model/flin/')
    parser.add_argument('--resource-dir', help='directory for storing resource', default='../resource/')
    parser.add_argument('--data-path', help='directory for dataset annotation', default='./dataset_dir/')
    parser.add_argument('--train-trace-id', help='train trace id', default='925')
    parser.add_argument('--test-trace-id', help='test trace id', default='925')
    parser.add_argument('--model-result-dict', help='model result dict', default='flin')

    parser.add_argument('--char-encoder-type', help='encoding model type [rnn]', default='rnn')
    parser.add_argument('--encoder-type', help='encoding model type [rnn_uni|rnn_bi]', default='rnn_bi')
    parser.add_argument('--train-mode', help='Training mode', default=False)
    parser.add_argument('--eval-mode', help='evaluation mode', default='of')
    parser.add_argument('--test-query-type', help='test query type', default='all')
    parser.add_argument('--ext-test-set', help='ext test set [d-flow|google]', default='d-flow')
    parser.add_argument('--ext-test-dom', help='ext test domain [R|H|S]', default='S')
    parser.add_argument('--verbose', help='verbose mode', default=True)

    args = vars(parser.parse_args())
    pp.pprint(args)

    main(args)
