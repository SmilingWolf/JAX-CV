import argparse
import json
import os

import numpy as np
import tensorflow as tf
from PIL import Image


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def load_existing_labels(label_mapping_filename):
    """Load existing label to index mapping from a file."""
    if os.path.exists(label_mapping_filename):
        with open(label_mapping_filename, 'r') as mapping_file:
            return json.load(mapping_file), True
    return {}, False

def save_label_mapping(label_mapping_filename, label_to_index):
    """Save label to index mapping to a file."""
    with open(label_mapping_filename, 'w') as mapping_file:
        json.dump(label_to_index, mapping_file, indent=4)

def create_tfrecord(dataset_folder, output_path, split_ratio=0.7, img_size=512):
    """Create a TFRecord file from images and label files and generate dataset JSON file."""
    dataset_name = os.path.basename(os.path.normpath(dataset_folder))
    train_writer = tf.io.TFRecordWriter(f'{output_path}/record_shards_train/{dataset_name}_train.tfrecord')
    val_writer = tf.io.TFRecordWriter(f'{output_path}/record_shards_val/{dataset_name}_val.tfrecord')
    
    image_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    label_files = [f for f in os.listdir(dataset_folder) if f.lower().endswith('txt')]
    
    image_files.sort()
    label_files.sort()
    
    # Create a set of image filenames without extensions for quick lookup
    image_file_set = set(os.path.splitext(f)[0].lower() for f in image_files)
    
    # Load existing label mapping
    label_mapping_filename = f'{output_path}/{dataset_name}_labels.json'
    label_to_index, mapping_exists = load_existing_labels(label_mapping_filename)
    index_to_label = [None] * (len(label_to_index) + 1)

    if mapping_exists:
        for label, index in label_to_index.items():
            index_to_label[index] = label

    # Collect new labels and update mapping
    num_samples = 0
    new_labels = set()

    for label_file in label_files:
        label_path = os.path.join(dataset_folder, label_file)
        image_name = os.path.splitext(label_file)[0].lower()

        # Check if there's a corresponding image file
        if image_name not in image_file_set:
            print(f"Skipping label file {image_name} because no corresponding image file found.")
            continue

        # Read labels and collect new labels
        with open(label_path, 'r') as f:
            labels = f.read().strip().split(', ')
            new_labels.update(labels)
    
    # Update label to index mapping
    for label in new_labels:
        if label not in label_to_index:
            new_index = len(label_to_index)
            label_to_index[label] = new_index
            index_to_label.append(label)
    
    # Create a set of label filenames (without extension) for quick lookup
    label_file_set = set(os.path.splitext(f)[0].lower() for f in label_files)
    
    # Number of unique tags
    num_classes = len(label_to_index)
    
    # Number of valid samples
    num_samples = len([f for f in image_files if os.path.splitext(f)[0].lower() in label_file_set])

    # Number of training and validation samples
    num_train_samples = int(num_samples * split_ratio)
    num_val_samples = num_samples - num_train_samples

    for idx, image_file in enumerate(image_files):
        image_name = os.path.splitext(image_file)[0].lower()
        label_file = f'{image_name}.txt'
        label_path = os.path.join(dataset_folder, label_file)
        
        # Check if there's a corresponding label file
        if image_name not in label_file_set:
            print(f"Skipping image file {image_name} because no corresponding label file was found.")
            continue

        # Read labels and convert to indices
        with open(label_path, 'r') as f:
            labels = f.read().strip().split(', ')
            label_indices = [label_to_index[label] for label in labels if label in label_to_index]

        # Read image
        image_path = os.path.join(dataset_folder, image_file)
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((img_size, img_size), Image.LANCZOS)
        image_np = np.array(image)
        # Convert RGB to BGR
        image_np = image_np[..., ::-1]
        
        image_id = hash(image_name) % 2**63
        
        # Create a feature
        feature = {
            'image_id': _int64_feature(image_id),
            'image_bytes': _bytes_feature(image_np),
            'label_indexes': _int64_list_feature(label_indices)
        }

        # Protocol buffers
        protobuf = tf.train.Example(features=tf.train.Features(feature=feature))
        serialized_example = protobuf.SerializeToString()

        # Write the buffer to the TFRecord files
        if idx < num_train_samples:
            train_writer.write(serialized_example)
        else:
            val_writer.write(serialized_example)

    train_writer.close()
    val_writer.close()

    dataset_info = {
        "num_classes": num_classes,
        "train_samples": num_train_samples,
        "val_samples": num_val_samples
    }

    json_filename = f'{output_path}/{dataset_name}.json'
    with open(json_filename, 'w') as json_file:
        json.dump(dataset_info, json_file, indent=4)
        
    # Save updated label to index mapping
    save_label_mapping(label_mapping_filename, label_to_index)

    print(f"TFRecord files saved to {output_path}/record_shards_train and {output_path}/record_shards_val")
    print(f"Dataset JSON file saved to {json_filename}")
    print(f"Label mapping JSON file saved to {label_mapping_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create TFRecord file from images and label files')
    parser.add_argument('--dataset_folder', type=str, help='Path to dataset folder containing both images and labels')
    parser.add_argument('--output_path', type=str, help='Path to output files. Will place TFRecords into "record_shards_train" and "record_shards_val" folders')
    parser.add_argument('--split_ratio', type=float, default=0.7, help='Ratio of training to total samples (default: 0.7)')
    parser.add_argument('--img_size', type=int, default=512, help='Image size to resize all images to (default: 512)')

    args = parser.parse_args()

    # Use dataset folder as output  if empty
    if args.output_path is None:
        args.output_path = args.dataset_folder

    create_tfrecord(args.dataset_folder, args.output_path, args.split_ratio, args.img_size)
