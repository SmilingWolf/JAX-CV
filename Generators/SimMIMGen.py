import numpy as np
import tensorflow as tf


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


class DataGenerator:
    def __init__(
        self,
        records_path,
        num_classes=2380,
        image_size=320,
        batch_size=32,
        num_devices=0,
        noise_level=0,
        mixup_alpha=0.2,
        rotation_ratio=0.25,
        cutout_max_pct=0.25,
        cutout_patches=1,
        random_resize_method=True,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6,
    ):
        """
        Noise level 1: augmentations I will never train without
                       unless I'm dealing with extremely small networks
                       (Random rotation, random cropping and random flipping)

        Noise level 2: more advanced stuff (MixUp)
        """

        self.records_path = records_path
        self.num_classes = num_classes
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_devices = num_devices
        self.noise_level = noise_level
        self.mixup_alpha = mixup_alpha
        self.rotation_ratio = rotation_ratio
        self.random_resize_method = random_resize_method

        self.cutout_max_pct = cutout_max_pct
        self.cutout_replace = 127
        self.cutout_patches = cutout_patches

        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

    def gen_mask(self, image, labels):
        rand_size = self.image_size // self.mask_patch_size
        scale = self.mask_patch_size // self.model_patch_size

        token_count = rand_size**2
        mask_count = tf.math.ceil(token_count * self.mask_ratio)
        mask_count = tf.cast(mask_count, tf.int32)

        mask_idx = tf.random.uniform((token_count,))
        mask_idx = tf.argsort(mask_idx)[:mask_count]

        mask = tf.reduce_max(tf.one_hot(mask_idx, token_count, dtype=tf.uint8), axis=0)
        mask = tf.reshape(mask, (rand_size, rand_size))
        mask = tf.repeat(mask, scale, axis=0)
        mask = tf.repeat(mask, scale, axis=1)
        return image, mask, labels

    def parse_single_record(self, example_proto):
        feature_description = {
            "image_id": tf.io.FixedLenFeature([], tf.int64),
            "image_bytes": tf.io.FixedLenFeature([], tf.string),
            "label_indexes": tf.io.VarLenFeature(tf.int64),
        }

        # Parse the input 'tf.train.Example' proto using the dictionary above.
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)
        image_tensor = tf.io.decode_jpeg(parsed_example["image_bytes"], channels=3)

        # RGB -> BGR (legacy reasons)
        image_tensor = tf.gather(image_tensor, axis=2, indices=[2, 1, 0])

        # Nel TFRecord mettiamo solo gli indici per questioni di spazio
        # Emula MultiLabelBinarizer a partire dagli indici per ottenere un tensor di soli 0 e 1
        label_indexes = tf.sparse.to_dense(
            parsed_example["label_indexes"],
            default_value=0,
        )
        one_hots = tf.one_hot(label_indexes, self.num_classes)
        labels = tf.reduce_max(one_hots, axis=0)
        labels = tf.cast(labels, tf.float32)

        return image_tensor, labels

    def random_flip(self, image, labels):
        image = tf.image.random_flip_left_right(image)
        return image, labels

    def random_crop(self, image, labels):
        image_shape = tf.shape(image)
        height = image_shape[0]
        width = image_shape[1]

        factor = tf.random.uniform(shape=[], minval=0.87, maxval=0.998)

        # Assuming this is a standard 512x512 Danbooru20xx SFW image
        new_height = new_width = tf.cast(tf.cast(height, tf.float32) * factor, tf.int32)

        offset_height = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=(height - new_height),
            dtype=tf.int32,
        )
        offset_width = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=(width - new_width),
            dtype=tf.int32,
        )
        image = tf.image.crop_to_bounding_box(
            image,
            offset_height,
            offset_width,
            new_height,
            new_width,
        )
        return image, labels

    def random_rotate(self, images, masks, labels):
        bs, h, w, c = tf.unstack(tf.shape(images))

        h = tf.cast(h, tf.float32)
        w = tf.cast(w, tf.float32)
        radians = np.pi * self.rotation_ratio
        radians = tf.random.uniform(shape=(bs,), minval=-radians, maxval=radians)

        cos_angles = tf.math.cos(radians)
        sin_angles = tf.math.sin(radians)
        x_offset = ((w - 1) - (cos_angles * (w - 1) - sin_angles * (h - 1))) / 2.0
        y_offset = ((h - 1) - (sin_angles * (w - 1) + cos_angles * (h - 1))) / 2.0
        zeros = tf.zeros((bs,), tf.float32)

        transforms = [
            cos_angles,
            -sin_angles,
            x_offset,
            sin_angles,
            cos_angles,
            y_offset,
            zeros,
            zeros,
        ]
        transforms = tf.transpose(transforms, (1, 0))
        images = tf.raw_ops.ImageProjectiveTransformV3(
            images=images,
            transforms=transforms,
            output_shape=[h, w],
            fill_value=255,
            interpolation="BILINEAR",
            fill_mode="CONSTANT",
        )
        return images, masks, labels

    def resize(self, image, labels):
        if self.random_resize_method:
            # During training mix algos up to make the model a bit more more resilient
            # to the different image resizing implementations out there (TF, OpenCV, PIL, ...)
            method_index = tf.random.uniform((), minval=0, maxval=3, dtype=tf.int32)
            if method_index == 0:
                image = tf.image.resize(
                    images=image,
                    size=(self.image_size, self.image_size),
                    method="area",
                    antialias=True,
                )
            elif method_index == 1:
                image = tf.image.resize(
                    images=image,
                    size=(self.image_size, self.image_size),
                    method="bilinear",
                    antialias=True,
                )
            else:
                image = tf.image.resize(
                    images=image,
                    size=(self.image_size, self.image_size),
                    method="bicubic",
                    antialias=True,
                )
        else:
            image = tf.image.resize(
                images=image,
                size=(self.image_size, self.image_size),
                method="area",
                antialias=True,
            )
        image = tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
        return image, labels

    # https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py
    def cutout(self, image, labels):
        """Apply cutout (https://arxiv.org/abs/1708.04552) to image.
        This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
        a random location within `img`. The pixel values filled in will be of the
        value `replace`. The located where the mask will be applied is randomly
        chosen uniformly over the whole image.
        Args:
          image: An image Tensor of type uint8.
          pad_size: Specifies how big the zero mask that will be generated is that
            is applied to the image. The mask will be of size
            (2*pad_size x 2*pad_size).
          replace: What pixel value to fill in the image in the area that has
            the cutout mask applied to it.
        Returns:
          An image Tensor that is of type uint8.
        """
        pad_pct = self.cutout_max_pct
        replace = self.cutout_replace

        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        img_area = image_height * image_width
        pad_area = tf.cast(img_area, dtype=tf.float32) * pad_pct
        pad_size = tf.cast(tf.math.sqrt(pad_area) / 2, dtype=tf.int32)

        # Sample the center location in the image where the zero mask will be applied.
        cutout_center_height = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=image_height,
            dtype=tf.int32,
        )

        cutout_center_width = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=image_width,
            dtype=tf.int32,
        )

        lower_pad = tf.maximum(0, cutout_center_height - pad_size)
        upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
        left_pad = tf.maximum(0, cutout_center_width - pad_size)
        right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

        cutout_shape = [
            image_height - (lower_pad + upper_pad),
            image_width - (left_pad + right_pad),
        ]
        padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
        mask = tf.pad(
            tf.zeros(cutout_shape, dtype=image.dtype),
            padding_dims,
            constant_values=1,
        )
        mask = tf.expand_dims(mask, -1)
        mask = tf.tile(mask, [1, 1, 3])
        image = tf.where(
            tf.equal(mask, 0),
            tf.ones_like(image, dtype=image.dtype) * replace,
            image,
        )
        return image, labels

    def mixup_single(self, images, masks, labels):
        alpha = self.mixup_alpha
        batch_size = tf.shape(images)[0]

        # Unpack one dataset, generate a second
        # by shuffling the input one on the batch axis
        images_one = tf.cast(images, tf.float32)
        labels_one = tf.cast(labels, tf.float32)

        idxs = tf.random.shuffle(tf.range(batch_size))
        images_two = tf.gather(images_one, idxs, axis=0)
        labels_two = tf.gather(labels_one, idxs, axis=0)

        # Sample lambda and reshape it to do the mixup
        l = sample_beta_distribution(batch_size, alpha, alpha)
        x_l = tf.reshape(l, (batch_size, 1, 1, 1))
        y_l = tf.reshape(l, (batch_size, 1))

        # Perform mixup on both images and labels by combining a pair of images/labels
        # (one from each dataset) into one image/label
        images = images_one * x_l + images_two * (1 - x_l)
        labels = labels_one * y_l + labels_two * (1 - y_l)

        images = tf.cast(tf.clip_by_value(images, 0, 255), tf.uint8)
        return images, masks, labels

    def genDS(self):
        files = tf.data.Dataset.list_files(self.records_path)
        files = files.cache()
        files = files.repeat()

        dataset = files.interleave(
            tf.data.TFRecordDataset,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False,
        )
        dataset = dataset.ignore_errors()
        dataset = dataset.shuffle(2 * self.batch_size)
        dataset = dataset.map(
            self.parse_single_record,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        if self.noise_level >= 1:
            dataset = dataset.map(self.random_flip, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(self.random_crop, num_parallel_calls=tf.data.AUTOTUNE)

        # Resize before batching. Especially important if random_crop is enabled
        dataset = dataset.map(self.resize, num_parallel_calls=tf.data.AUTOTUNE)

        if self.noise_level >= 2 and self.cutout_max_pct > 0.0:
            for _ in range(self.cutout_patches):
                dataset = dataset.map(self.cutout, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.map(self.gen_mask, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(
            self.batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # Rotation is very slow on CPU. Rotating a batch of resized images is much faster
        if self.noise_level >= 1 and self.rotation_ratio > 0.0:
            dataset = dataset.map(
                self.random_rotate,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        if self.noise_level >= 2 and self.mixup_alpha > 0.0:
            dataset = dataset.map(
                self.mixup_single,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        if self.num_devices > 0:
            dataset = dataset.batch(
                self.num_devices,
                drop_remainder=True,
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        dataset = dataset.map(
            lambda images, masks, labels: (
                {
                    "images": tf.cast(images, tf.float32) * (1.0 / 127.5) - 1,
                    "masks": masks,
                    "labels": labels,
                }
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
