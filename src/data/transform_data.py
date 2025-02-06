from tqdm.auto import tqdm
import os
import pandas as pd

PATH = "/scratch/perez/double_liquidation"

def main():
    pbar = tqdm(os.listdir(PATH))
    for folder in pbar:
        pbar.set_description(folder)
        path = os.path.join(PATH, folder)

        for name in tqdm(os.listdir(path)):
            if name[-3:] != "csv":
                continue

            new_name = name[:-3] + "h5"
            df = pd.read_csv(os.path.join(path, name))
            df.to_hdf(os.path.join(path, new_name), key="df")


if __name__ == "__main__":
    main()
