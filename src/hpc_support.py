import dask.dataframe as dd

def process_large_dataset_with_dask(data_path):
    # Load large dataset using Dask for out-of-core computing
    ddf = dd.read_csv(data_path)

    # Example: Compute the mean of each column
    means = ddf.mean().compute()
    print(means)

if __name__ == "__main__":
    process_large_dataset_with_dask('data/large_dataset.csv'