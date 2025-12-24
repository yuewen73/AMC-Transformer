# train.py
import argparse
import numpy as np
import tensorflow as tf

from dataset import RadioML18Dataset
from model import create_vit_classifier


def main(args):
    # -----------------------------
    # Load dataset
    # -----------------------------
    train_ds = RadioML18Dataset(
        data_dir=args.data_dir,
        mode="train",
        seed=args.seed,
    )
    valid_ds = RadioML18Dataset(
        data_dir=args.data_dir,
        mode="valid",
        seed=args.seed,
    )
    test_ds = RadioML18Dataset(
        data_dir=args.data_dir,
        mode="test",
        seed=args.seed,
    )

    num_classes = len(train_ds.target_modulations)

    # Optional: SNR filtering for faster laptop runs
    if args.snr is not None:
        train_mask = train_ds.Z == args.snr
        valid_mask = valid_ds.Z == args.snr
        test_mask  = test_ds.Z  == args.snr

        train_ds.X, train_ds.Y = train_ds.X[train_mask], train_ds.Y[train_mask]
        valid_ds.X, valid_ds.Y = valid_ds.X[valid_mask], valid_ds.Y[valid_mask]
        test_ds.X,  test_ds.Y  = test_ds.X[test_mask],  test_ds.Y[test_mask]

    # Optional: subsample for speed
    if args.max_samples > 0:
        train_ds.X = train_ds.X[:args.max_samples]
        train_ds.Y = train_ds.Y[:args.max_samples]

    # Reshape for ViT
    x_train = train_ds.X.transpose(0, 2, 1).reshape(-1, 2, 1024, 1)
    y_train = tf.keras.utils.to_categorical(train_ds.Y, num_classes)

    x_val = valid_ds.X.transpose(0, 2, 1).reshape(-1, 2, 1024, 1)
    y_val = tf.keras.utils.to_categorical(valid_ds.Y, num_classes)

    x_test = test_ds.X.transpose(0, 2, 1).reshape(-1, 2, 1024, 1)
    y_test = tf.keras.utils.to_categorical(test_ds.Y, num_classes)

    # -----------------------------
    # Build model
    # -----------------------------
    model = create_vit_classifier(
        transformer_layers=args.layers,
        num_heads=args.heads,
        learning_rate=args.lr,
    )

    # -----------------------------
    # Train
    # -----------------------------
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    # -----------------------------
    # Evaluate
    # -----------------------------
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    # Model hyperparameters (match README)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--heads", type=int, default=8)

    # Training setup
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Speed-up options
    parser.add_argument("--snr", type=int, default=None,
                        help="Use a single SNR (e.g., 10) for fast reproduction")
    parser.add_argument("--max_samples", type=int, default=5000,
                        help="Limit number of training samples (CPU-friendly)")

    args = parser.parse_args()
    main(args)
