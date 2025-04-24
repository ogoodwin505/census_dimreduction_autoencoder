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
    import yaml
    from torchgeodemo import autoencoder_train_latent
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

    #save the data
    data_table.to_csv("data/uk_census_all.csv")
    return (
        Isomap,
        PCA,
        autoencoder_train_latent,
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
        yaml,
    )


@app.cell
def _(os, yaml):
    bottleneck_sizes = [8]

    # Define the base YAML structure as a dictionary
    BASE_YAML = {
        "data": {
            "source": "data/uk_census_all.csv",
            "nickname": "census_geodemo",
            "id_col": "OA"
        },
        "working_dir": "ukcensus/bottleneck_size_scan",
        "random_seed": 20210321,
        "autoencoder": {
            "nickname": "bottleneck_PLACEHOLDER",
            "version": "0_2",
            "save_latent": "csv",
            "max_epochs": 60,
            "batch_size": 0.01,
            "encoder": {
                "sizes": [256, 128,96],
                "activation": "LeakyReLU",
                "sparse": {
                    "topk_k": "PLACEHOLDER",
                    "sparsity_loss_weight": 0.01
                }
            },
            "decoder": {
                "sizes": [96, 128,256],
                "activation": "LeakyReLU"
            }
        }
    }

    # Output directory for generated YAMLs
    OUTPUT_DIR = "ukcensus/yamls"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate YAML files with varying latent space sizes
    for _latent_size in bottleneck_sizes:
        yaml_config = BASE_YAML.copy()
        yaml_config["autoencoder"]["nickname"] = f"bottleneck_{_latent_size}"
        yaml_config["autoencoder"]["encoder"]["sparse"]["topk_k"] = _latent_size

        config_path = os.path.join(OUTPUT_DIR, f"config_{_latent_size}.yaml")
        with open(config_path, "w") as yaml_file:
            yaml.dump(yaml_config, yaml_file, default_flow_style=False)

    print(f"Generated YAML configurations in {OUTPUT_DIR}")
    return (
        BASE_YAML,
        OUTPUT_DIR,
        bottleneck_sizes,
        config_path,
        yaml_config,
        yaml_file,
    )


@app.cell
def _(autoencoder_train_latent, bottleneck_sizes):

    for _latent_size in bottleneck_sizes:
        yaml_path = f"ukcensus/yamls/config_{_latent_size}.yaml"
        autoencoder_train_latent.main(yaml_path, create_latent=False)
    return (yaml_path,)


@app.cell
def _(PCA, data_table):
    n_features = data_table.shape[1]
    data = data_table

    # Fit PCA once with the maximum number of components
    pca = PCA(n_components=n_features - 1)
    transformed = pca.fit_transform(data)
    return data, n_features, pca, transformed


@app.cell
def _(bottleneck_sizes, data, np, pca, pd, plt, tqdm, transformed):
    # Lists to store RMSE and MAE
    rmse_pca, rmse_ae = [], []
    mae_pca, mae_ae = [], []

    # Iterate over different bottleneck sizes
    for n in tqdm(bottleneck_sizes, desc="Computing errors"):
        # PCA reconstruction
        reconstructed_pca = (transformed[:, :n] @ pca.components_[:n]) + pca.mean_
        error_pca = data - reconstructed_pca
        rmse_pca.append(np.sqrt(np.mean(error_pca ** 2)))
        mae_pca.append(np.mean(np.abs(error_pca)))

        # Autoencoder reconstruction
        reconstructed_ae = pd.read_csv(
            f"ukcensus/bottleneck_size_scan/census_geodemo__bottleneck_{n}_v0_2__reconstructed_outputs.csv", 
            index_col="ID"
        )
        error_ae = data - reconstructed_ae
        rmse_ae.append(np.sqrt(np.mean(error_ae ** 2)))
        mae_ae.append(np.mean(np.abs(error_ae)))

    # Plot RMSE
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(bottleneck_sizes, rmse_pca, marker='o', label="PCA")
    plt.plot(bottleneck_sizes, rmse_ae, marker='s', label="Autoencoder")
    plt.xlabel("Number of Components / Bottleneck Size")
    plt.ylabel("RMSE")
    plt.title("RMSE vs. # of Dimensions")
    plt.legend()
    plt.grid()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(bottleneck_sizes, mae_pca, marker='o', label="PCA")
    plt.plot(bottleneck_sizes, mae_ae, marker='s', label="Autoencoder")
    plt.xlabel("Number of Components / Bottleneck Size")
    plt.ylabel("MAE")
    plt.title("MAE vs. # of Dimensions")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
    return (
        error_ae,
        error_pca,
        mae_ae,
        mae_pca,
        n,
        reconstructed_ae,
        reconstructed_pca,
        rmse_ae,
        rmse_pca,
    )


if __name__ == "__main__":
    app.run()
