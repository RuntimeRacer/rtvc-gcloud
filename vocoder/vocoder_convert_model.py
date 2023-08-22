import argparse

from vocoder.wavernn.libwavernn.convert import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts a trained vocoder model for libwavernn.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Arguments
    parser.add_argument("model_fpath", type=str, help="Path to the model")
    parser.add_argument("--default_model_type", type=str, default=base.MODEL_TYPE_FATCHORD, help="default model type")
    parser.add_argument("--out_dir", type=str, default="vocoder/libwavernn/models/", help="Path to the output file")
    args = parser.parse_args()

    # Process the arguments
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(exist_ok=True)

    # Run the conversion
    convert_model(**vars(args))



