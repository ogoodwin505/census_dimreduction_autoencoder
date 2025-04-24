import marimo

__generated_with = "0.11.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    from itertools import combinations
    from tqdm import tqdm
    import os
    import re
    #notebook to rest reconstruction error of PCA

    #load the census data
    def populate_data_table(filepath: str) -> pd.DataFrame:
        print("Populating data table from files in", filepath)
        data_table = None  # Initialize data_table outside the loop

        for file in tqdm(os.listdir(filepath), desc="Processing files", unit="file"):
            df = pd.read_csv(filepath + "/" + file, index_col="OA")
            if data_table is None:
                data_table = df  # Initialize data_table with the first file encountered
            else:
                # Merge on index
                data_table = data_table.merge(df, left_index=True, right_index=True, how="outer")
        return data_table

    # Load the data
    data_path = "data/uk_census/"
    data_table = populate_data_table(data_path)

    #drop the totals (columns ending 001)
    data_table = data_table.loc[:, ~data_table.columns.str.endswith("001")]
    return (
        Isomap,
        PCA,
        combinations,
        data_path,
        data_table,
        mo,
        np,
        os,
        pd,
        pl,
        plt,
        populate_data_table,
        re,
        tqdm,
    )


@app.cell
def _(data_table):
    #save to csv
    data_table.to_csv("data/uk_census_data_fracoftotal.csv")

    return


@app.cell
def _(PCA, data_table, np, plt, tqdm):
    n_features = data_table.shape[1]
    data = data_table

    # Fit PCA once with the maximum number of components
    pca = PCA(n_components=n_features - 1)
    transformed = pca.fit_transform(data)

    # Compute reconstruction error for different numbers of components
    reconstruction_errors = []
    n_components_range = range(1, n_features)

    for n in tqdm(n_components_range, desc="Computing reconstruction errors"):
        reconstructed = (transformed[:, :n] @ pca.components_[:n]) + pca.mean_  # Use only first n components
        error = np.mean((data - reconstructed) ** 2)
        reconstruction_errors.append(error)

    # Plot reconstruction error
    plt.plot(n_components_range, reconstruction_errors, marker='o')
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Reconstruction Error")
    plt.title("PCA Reconstruction Error vs. Number of Components")
    plt.grid()
    plt.show()
    return (
        data,
        error,
        n,
        n_components_range,
        n_features,
        pca,
        reconstructed,
        reconstruction_errors,
        transformed,
    )


@app.cell
def _(os, pd, re):
    torch_data_path = "/home/ogoodwin/torchgeodemo-pkg-dev/ukcensus/bottleneck_size_scan/"

    bottleneck_sizes = []
    reco_errs = []

    for file in os.listdir(torch_data_path):
        if file.endswith(".csv"):
            match = re.search(r"bottleneck_(\d+)", file)
            if match:
                bottleneck_size = int(match.group(1))
                torch_data = pd.read_csv(os.path.join(torch_data_path, file))
                reco_errs.append(torch_data["Reconstruction_Error"].mean())
                bottleneck_sizes.append(bottleneck_size)

    # Convert to DataFrame if needed
    df = pd.DataFrame({"Bottleneck_Size": bottleneck_sizes, "Mean_Reco_Error": reco_errs})
    #sort by bottleneck size
    df = df.sort_values("Bottleneck_Size")
    print(df)


    #fgdfg
    return (
        bottleneck_size,
        bottleneck_sizes,
        df,
        file,
        match,
        reco_errs,
        torch_data,
        torch_data_path,
    )


@app.cell
def _(df, plt):
    plt.plot(df["Bottleneck_Size"], df["Mean_Reco_Error"], marker='o')
    return


@app.cell
def _():
    # # Compute reconstruction error when dropping each component individually
    # drop_errors = []
    # pca_full = PCA(n_components=n_features)
    # pca_full.fit(data)
    # transformed_full = pca_full.transform(data)

    # for i in tqdm(range(n_features), desc="Computing drop errors"):
    #     transformed_dropped = transformed_full.copy()
    #     transformed_dropped[:, i] = 0  # Zero out one component at a time
    #     reconstructed_dropped = pca_full.inverse_transform(transformed_dropped)
    #     _error = np.mean((data - reconstructed_dropped) ** 2)
    #     drop_errors.append(_error)

    # # Plot drop reconstruction error
    # plt.bar(range(1, n_features+1), drop_errors)
    # plt.xlabel("Dropped PCA Component")
    # plt.ylabel("Reconstruction Error")
    # plt.title("Reconstruction Error When Dropping Each PCA Component Individually")
    # plt.show()

    # # Test if drop errors are monotonically decreasing
    # is_monotonic = all(drop_errors[i] >= drop_errors[i + 1] for i in range(len(drop_errors) - 1))
    # print("Drop errors are monotonically decreasing:", is_monotonic)
    # #if not print the drop error which is not monotonically decreasing
    # if not is_monotonic:
    #     print("First non-monotonic drop error:", next(drop_errors[i] for i in range(len(drop_errors) - 1) if drop_errors[i] < drop_errors[i + 1]))
    return


if __name__ == "__main__":
    app.run()
