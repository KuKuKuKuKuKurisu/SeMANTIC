import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='Dialogue-Generation Research')

    parser.add_argument(
        "--model_file_name",
        type=str,
        default='dst_aware.pth'
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default='0'
    )

    return parser.parse_args()
