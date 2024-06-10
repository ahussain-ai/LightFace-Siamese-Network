
import numpy  as np
from pairs import *
import tensorflow as tf


def create_dataset(dataset_dir = None, batch_size = 64,num_pairs = 10000) :

    if not dataset_d :
        print(f"eneter dataset path")
        return 

    dataset_d = arrange_dataset(dataset_dir)
    (pos,pos_y),(neg,neg_y), _  = generate_pairs(dataset_d, num_pairs = num_pairs)
    print(f"total positive sample {len(pos)}")
    print(f"Total negative sample {len(neg)}")

    pos_1 = np.array([ pair[0] for pair in pos ])
    pos_2 = np.array([ pair[1] for pair in pos ])

    neg_1 = np.array([ pair[0] for pair in neg ])
    neg_2 = np.array([ pair[1] for pair in neg]) 

    # Combine anchor and negative data
    all_images_1 = np.concatenate([pos_1, neg_1], axis=0)
    all_images_2 = np.concatenate([pos_2, neg_2], axis=0)
    all_labels = np.concatenate([pos_y, neg_y], axis=0)

    # Create a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(((all_images_1, all_images_2), all_labels))

    dataset = dataset.shuffle(len(all_labels))
    dataset = dataset.map(preprocess_image)


    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    train_dataset = dataset.take(train_size).batch(batch_size)
    test_dataset = dataset.skip(train_size).take(test_size).batch(batch_size)

    print(f"train dataset {len(train_dataset)*64}")
    print(f"test datset {len(test_dataset)*32}")

    return train_dataset, test_dataset

if __name__ == '__main__' : 
    pass

