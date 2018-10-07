from PIL import Image
import jieba
import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
import os
import argparse
import gensim.models.word2vec as w2v
import numpy as np

parser = argparse.ArgumentParser()
def default_loader(path):
    return Image.open(path).convert('RGB')


class WordSegmentation(object):
    def __init__(self):
        self.sentence = ""

    def set_sentence(self, sentence=""):
        self.sentence = sentence

    def is_chinese(self, uchar):
        """判断一个unicode是否是汉字"""
        if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
            return True
        else:
            return False

    def is_number(self, uchar):
        """判断一个unicode是否是数字"""
        if uchar >= u'\u0030' and uchar <= u'\u0039':
            return True
        else:
            return False

    def is_alphabet(self, uchar):
        """判断一个unicode是否是英文字母"""
        if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
            return True
        else:
            return False

    def is_legal(self, uchar):
        """判断是否非汉字，数字和英文字符"""
        # if not (self.is_chinese(uchar) or self.is_number(uchar) or self.is_alphabet(uchar)):
        if not (self.is_chinese(uchar) or self.is_alphabet(uchar)):
            return False
        else:
            return True

    def extract_chinese(self, line):
        res = ""
        for word in line:
            if self.is_legal(word):
                res = res + word
        return res

    def run_segmentation(self):
        line = self.extract_chinese(self.sentence)
        self._words = jieba.lcut(line, cut_all=False, HMM=True)

    def words(self):
        return self._words
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class MyImageFloder(data.Dataset):
    def __init__(self, images_dir, loader=default_loader, transform=None, target_transform=None):

        classes, class_to_idx = find_classes(images_dir)
        images = []
        for target in sorted(os.listdir(images_dir)):
            d = os.path.join(images_dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)


        self.images_dir = images_dir
        self.loader = loader

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.images = images

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.images[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, [target,path]

    def __len__(self):
        return len(self.images)

class DataLoader(object):
    def __init__(self,paras):
        """
        :param paras: 数据参数
         paras = {
            "images_dir": self.images_dir,
            "first_category": self.first_category,
            "texts_dir": self.texts_dir,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size
        }
        """
        self._images_dir = paras["images_dir"]
        self._first_category = paras["first_category"]
        self._texts_dir = paras["texts_dir"]
        self.num_workers = paras["num_workers"]
        self.batch_size = paras["batch_size"]

        # images loader
        self.images_dir = os.path.join(self._images_dir,self._first_category)
        self._init_img_data_transforms()
        self._init_img_data_loaders()

        # texts info loader
        self.texts_dir = os.path.join(self._texts_dir, self._first_category)
        self._text_all_words_file = os.path.join(self.texts_dir, "all_words.txt") # 当前类目下的所有词库
        self._text_blank_file = os.path.join(self._texts_dir,"blank_file.txt")
        if not os.path.exists(self._text_blank_file):
            with open(self._text_blank_file, 'w') as file: pass

        self._text_labels_index = {}
        self._text_index_labels = {}
        self.fs = os.listdir(self.texts_dir)
        self._TEXT_MAX_SEQUENCE_LENGTH = 30
        self._TEXT_MAX_NB_WORDS  = 12888
        self._TEXT_EMBEDDING_DIM = 20

        self.word_segmentation = WordSegmentation()

        for i, f in enumerate(sorted(self.fs)):  # 遍历每一个类文件夹
            if os.path.isfile(os.path.join(self.texts_dir, f)): continue
            self._text_labels_index[f] = i
            self._text_index_labels[i] = f

        self._init_text_basic_variables()       # 初始化词表等，并将每个 first_category 中的词保存下来
        self._init_text_w2v()                   # 对 texts_dir 中的文字训练词向量
        self._init_text_embedding_matrix()      # 生成词表的 embedding_matrix
        self._text_turn_words2index()           # 生成训练数据，需要将训练数据的Word转换为word的索引
    # Image Dataloaders
    def _init_img_data_transforms(self):
        # Data augmentation and normalization for training
        # Just normalization for validation
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

    def _init_img_data_loaders(self):
        self.image_datasets = {x: MyImageFloder(images_dir=os.path.join(self.images_dir, x), transform=self.data_transforms[x])
                               for x in ['train', 'val']}
        self.dataloaders = {x: torch.utils.data.DataLoader(
            self.image_datasets[x], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers) for x in
            ['train', 'val']}

    # Texts Info DataLoaders
    def load_texts(self):
        num_recs = 0
        texts = []
        labels = []  # list of label ids
        for la in self._text_labels_index.keys(): # 每个类别文件夹
            if "train" in la or "val" in la: continue
            la_dir = os.path.join(self.texts_dir, la)
            fs = os.listdir(la_dir)
            for f in fs:                    # 每个类别里面的文件
                with open(la_dir + "/" + f, encoding='utf-8') as file:
                    text = file.read()
                    self.word_segmentation.set_sentence(text)
                    self.word_segmentation.run_segmentation()
                    text_info = self.word_segmentation.words()
                    texts.append(text_info)
                    labels.append(self._text_labels_index[la])
                    num_recs = num_recs + 1
        if not os.path.exists(self._text_all_words_file):
            # 将分词后的 texts 存储到相应文件夹下
            with open(self._text_all_words_file, 'w') as file:
                file.writelines("\n".join([" ".join(line) for line in texts]))

        return texts, labels, self._text_labels_index, self._text_index_labels

    def _init_text_w2v(self):
        print("training word2vec...")
        if not os.path.exists(self._text_all_words_file.replace(".txt", ".w2v")):
            sentences = w2v.LineSentence(self._text_all_words_file)
            model = w2v.Word2Vec(sentences, size=20, window=5, min_count=5, workers=4)
            model.save(self._text_all_words_file.replace(".txt", ".w2v"))

    def _init_text_basic_variables(self):
        self.texts, self.labels, self.labels_index, self.index_labels = self.load_texts()

        # 词表
        self.word_vocb = []
        self.word_vocb.append('')
        for text in self.texts:
            for word in text:
                self.word_vocb.append(word)
        self.word_vocb = set(self.word_vocb)
        self.vocb_size = len(self.word_vocb)

        # 设置词表大小
        self.nb_words = self.vocb_size
        self.max_len = self._TEXT_MAX_SEQUENCE_LENGTH
        self.word_dim = self._TEXT_EMBEDDING_DIM
        self.n_class = len(self.index_labels)

    def _init_text_embedding_matrix(self):
        # 词表与索引的map
        self.word_to_idx = {word: i for i, word in enumerate(self.word_vocb)}
        self.idx_to_word = {self.word_to_idx[word]: word for word in self.word_to_idx}
        # 每个单词的对应的词向量
        embeddings_index = self.load_text_w2v()
        # 预先处理好的词向量
        embedding_matrix = np.zeros((self.nb_words, self.word_dim))
        for word, i in self.word_to_idx.items():
            if i >= self.nb_words:
                continue
            if word in embeddings_index:
                embedding_vector = embeddings_index[word]
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
        self.embedding_matrix = embedding_matrix
    def _text_turn_words2index(self):
        # 生成训练数据，需要将训练数据的Word转换为word的索引
        self.texts_with_id = np.zeros([len(self.texts), self.max_len])

        for i in range(0, len(self.texts)):
            if len(self.texts[i]) < self.max_len:
                for j in range(0, len(self.texts[i])):
                    self.texts_with_id[i][j] = self.word_to_idx[self.texts[i][j]]
                for j in range(len(self.texts[i]), self.max_len):
                    self.texts_with_id[i][j] = self.word_to_idx['']
            else:
                for j in range(0, self.max_len):
                    self.texts_with_id[i][j] = self.word_to_idx[self.texts[i][j]]

    def find_text_path(self,image_path):
        elements = image_path.split("/")
        image_id = elements[-1].replace(".jpg","")
        image_target = elements[-2]
        # train_or_test = elements[-3] 实际上文本数据不需要划分 train or val
        text_path = os.path.join(self.texts_dir, image_target, image_id)
        if os.path.exists(text_path + ".txt"):
            return text_path + ".txt"
        else:
            return self._text_blank_file
    def load_texts_batch(self,images_path):
        """
        根据 images_path 找到对应的文本信息并返回
        :param images_path,当前训练所用的batch data: [tuple] (path1, path2, ...)
        :return:
        """
        texts = []
        for image_path in images_path:
            image_text_path = self.find_text_path(image_path)
            with open(image_text_path,encoding='utf-8') as file:
                text = file.read()
                self.word_segmentation.set_sentence(text)
                self.word_segmentation.run_segmentation()
                text_info = self.word_segmentation.words()
                texts.append(text_info)
        # word2vec
        input_text = np.zeros([len(images_path), self.max_len])
        for i,text in enumerate(texts):
            if len(text) > self.max_len:
                for j,word in enumerate(text):
                    if j >= self.max_len: continue
                    input_text[i][j] = self.word_to_idx[word]
            else:
                for j in range(0, len(text)):
                    input_text[i][j] = self.word_to_idx[text[j]]
                for j in range(len(text), self.max_len):
                    input_text[i][j] = self.word_to_idx['']
        return input_text

    def load_text_w2v(self):
        """
        读入每个word 的embedding向量
        :return:
        """
        if not os.path.exists(self._text_all_words_file.replace(".txt", ".w2v")):
            sentences = w2v.LineSentence(self._text_all_words_file)
            model = w2v.Word2Vec(sentences, size=20, window=5, min_count=5, workers=4)
            model.save(self._text_all_words_file.replace(".txt", ".w2v"))
        else:
            model = w2v.Word2Vec.load(self._text_all_words_file.replace(".txt", ".w2v"))
        return model
if __name__ == '__main__':
    pass