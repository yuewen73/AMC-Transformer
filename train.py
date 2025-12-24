# train.py
import argparse
import numpy as np
import tensorflow as tf

from dataset import RadioML18Dataset
from model import create_vit_classifier


def main(args):
    train_ds = RadioML18Dataset(args.data_dir, "train", args.seed)
    valid_ds = RadioML18Dataset(args.data_dir, "valid", args.seed)
    test_ds  = RadioML18Dataset(args.data_dir, "test",  args.seed)

    num_classes = len(train_ds.target_modulations)

    x_train = train_ds.X.transpose(0, 2, 1).reshape(-1, 2, 1024, 1)
    y_train = tf.keras.utils.to_categorical(train_ds.Y, num_classes)
    x_val   = valid_ds.X.transpose(0, 2, 1).reshape(-1, 2, 1024, 1)
    y_val   = tf.keras.utils.to_categorical(valid_ds.Y, num_classes)
    x_test  = test_ds.X.transpose(0, 2, 1).reshape(-1, 2, 1024, 1)
    y_test  = tf.keras.utils.to_categorical(test_ds.Y, num_classes)

    model = create_vit_classifier(
        transformer_layers=args.layers,
        num_heads=args.heads,
    )

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=512)
    args = p.parse_args()

    main(args)
