import argparse
from methods.sarad.wrapper import train as sarad_train

def main(args):
    """
    Entry point for SARAD baseline in our pipeline system.
    """

    print("[PIPELINE] Running SARAD baseline")

    sarad_train(
        use_processed=args.use_processed,
        processed_path=args.processed_path,
        dataset=args.dataset,
        device=args.device,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_processed", type=bool, default=True)
    parser.add_argument("--processed_path", type=str, help="Input the full path of the parent directory of where the dataset is included", default="D:\Sreya\Case_Western\OODResearch\ood-quantcwru\datasets\\raw")
    parser.add_argument("--dataset", type=str, default="SMD")
    parser.add_argument("--device", type=str, default="gpu")

    args = parser.parse_args()
    main(args)