import argparse
from methods.sarad.wrapper import train as sarad_train

def main(args):
    """
    Entry point for SARAD baseline in our pipeline system.
    """

    print("[PIPELINE] Running SARAD baseline")

    sarad_train(
        dataset=args.dataset,
        device=args.device,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SMD")
    parser.add_argument("--device", type=str, default="gpu")

    args = parser.parse_args()
    main(args)