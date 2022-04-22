class Parameters(object):
    # num_epochs = 12
    num_epoch = 5
    batch_size = 128
    tag_nums = 13  # 标签数目
    # hidden_nums = 650  # bi-lstm的隐藏层单元数目
    hidden_nums = 128
    # learning_rate = 0.1
    learning_rate = 0.001
    # dropout_keep_prob = 1.5
    dropout_keep_prob = 0.5
    sentence_length = 25
    clip = 5.0
    lr = 0.9



    #train_data_path = "./data_ai/nerData/80000-data.txt"
    train_data_path = "./data_ai/nerData/train.txt"
    #train_label_path = "./data_ai/nerData/80000_data_label.txt"
    train_label_path = "./data_ai/nerData/label.txt"

    test_data_path = "./data_ai/nerData/80000-test-data.txt"
    test_label_path = "./data_ai/nerData/80000_test_data_label.txt"
    word_embedings_path = "./data_ai/nerData/word_vec_128.model"
    vocb_path = "./data_ai/nerData/word128.vab"
    #model_checkpoint_path ="./data_ai/nerModel"
    model_checkpoint_path = "./data_ai/nerModel/bilstm-crf.models-2222"
    #model_checkpoint_path="./data_ai/nerModel/"

    # train_data_path = "./data_ai/nerData/train_data采用的字向量.txt"
    # train_label_path = "./data_ai/nerData/train_label.txt"
    # test_data_path = "./data_ai/nerData/test_data.txt"
    # test_label_path = "./data_ai/nerData/test_label.txt"
    # word_embedings_path = "./data_ai/cbowData/ner_model1.model"
    # vocb_path = "./data_ai/cbowData/ner1.vab"
    # model_checkpoint_path ="./data_ai/nerModel/bilstm-crf.models-2254"