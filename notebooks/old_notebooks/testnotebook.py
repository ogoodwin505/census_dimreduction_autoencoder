import marimo

__generated_with = "0.11.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    #generate a 1000 point 8 dimensional dataset, with some structure

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import Isomap
    from itertools import combinations
    #notebook to rest reconstruction error of PCA
    return Isomap, PCA, combinations, mo, np, pl, plt


@app.cell
def _(PCA, np, plt):
    n_features=20

    # Generate a non-linear dataset (e.g., Swiss roll)
    def generate_nonlinear_data(n_samples=10000, n_features=n_features, noise=0.3):
        t = 3 * np.pi * (np.random.rand(n_samples) - 0.5)  # Non-linear structure
        x = t * np.cos(t)
        y = t * np.sin(t)
        z = np.random.randn(n_samples) * noise  # Adding some noise
    
        # Generate additional features with some dependency on the main structure
        additional_features = np.random.randn(n_samples, n_features - 3)
        additional_features[:, 0] = np.sin(t)  # Introduce some correlation
        additional_features[:, 1] = np.cos(t)
        additional_features[:, 2] = t
    
        data = np.column_stack([x, y, z, additional_features])
        return data

    # Generate dataset
    data = generate_nonlinear_data()

    # Compute reconstruction error for different PCA components
    reconstruction_errors = []
    n_components_range = range(1, n_features)

    for n in n_components_range:
        pca = PCA(n_components=n)
        transformed = pca.fit_transform(data)
        reconstructed = pca.inverse_transform(transformed)
        error = np.mean((data - reconstructed) ** 2)
        reconstruction_errors.append(error)
        explained_variance = pca.explained_variance_ratio_.sum()

    # Plot reconstruction error
    plt.plot(n_components_range, reconstruction_errors, marker='o')
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Reconstruction Error")
    plt.title("PCA Reconstruction Error vs. Number of Components")
    plt.grid()
    plt.show()

    #plot the variance explained by each component
    plt.plot(n_components_range, np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance vs. Number of Components")
    plt.grid()
    plt.show()


    return (
        data,
        error,
        explained_variance,
        generate_nonlinear_data,
        n,
        n_components_range,
        n_features,
        pca,
        reconstructed,
        reconstruction_errors,
        transformed,
    )


@app.cell
def _(PCA, data, n_features, np, plt):
    # Compute reconstruction error when dropping each component individually


    drop_errors = []
    pca_full = PCA(n_components=n_features)
    pca_full.fit(data)
    transformed_full = pca_full.transform(data)

    for i in range(n_features):
        transformed_dropped = transformed_full.copy()
        transformed_dropped[:, i] = 0  # Zero out one component at a time
        reconstructed_dropped = pca_full.inverse_transform(transformed_dropped)
        _error = np.mean((data - reconstructed_dropped) ** 2)
        drop_errors.append(_error)

    # Plot drop reconstruction error
    plt.bar(range(1, n_features+1), drop_errors)
    plt.xlabel("Dropped PCA Component")
    plt.ylabel("Reconstruction Error")
    plt.title("Reconstruction Error When Dropping Each PCA Component Individually")
    plt.show()

    plt.bar(range(1, n_features+1), pca_full.explained_variance_ratio_)
    plt.xlabel("PCA Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Explained Variance Ratio for Each PCA Component")
    plt.show()


    plt.plot(drop_errors,pca_full.explained_variance_ratio_, 'o')
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Explained Variance Ratio")
    plt.title("Explained Variance Ratio vs. Reconstruction Error")
    plt.show()
    #test if drop errors are monotonically decreasing
    all(drop_errors[i] >= drop_errors[i + 1] for i in range(len(drop_errors) - 1))
    return (
        drop_errors,
        i,
        pca_full,
        reconstructed_dropped,
        transformed_dropped,
        transformed_full,
    )


@app.cell
def _(combinations, data, n_features, np, pca_full, plt, transformed_full):

    def _pair_test():
        # Compute reconstruction error when dropping random pairs of components
        pair_drop_errors = {}
        component_pairs = list(combinations(range(n_features), 2))

        for pair in component_pairs:
            transformed_dropped = transformed_full.copy()
            transformed_dropped[:, pair[0]] = 0
            transformed_dropped[:, pair[1]] = 0
            reconstructed_dropped = pca_full.inverse_transform(transformed_dropped)
            error = np.mean((data - reconstructed_dropped) ** 2)
            pair_drop_errors[pair] = error

        # Find the best pair to drop
        best_pair = min(pair_drop_errors, key=pair_drop_errors.get)
        print(f"Best pair to drop: {best_pair} with error: {pair_drop_errors[best_pair]}")

        # Compare with dropping the last two components
        last_two = (n_features-2, n_features-1)
        transformed_dropped = transformed_full.copy()
        transformed_dropped[:, last_two[0]] = 0
        transformed_dropped[:, last_two[1]] = 0
        reconstructed_dropped = pca_full.inverse_transform(transformed_dropped)
        last_two_error = np.mean((data - reconstructed_dropped) ** 2)
        print(f"Error when dropping last two components: {last_two_error}")
            # Plot reconstruction error for each pair
        pairs, errors = zip(*pair_drop_errors.items())
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(errors)), errors, tick_label=[str(p) for p in pairs], color='skyblue')
        plt.xticks(rotation=90)
        plt.xlabel("Dropped PCA Component Pairs")
        plt.ylabel("Reconstruction Error")
        plt.title("Reconstruction Error When Dropping Component Pairs")
        plt.show()

    _pair_test()
    return


@app.cell
def _(combinations, data, n_features, np, pca_full, plt, transformed_full):
    def _trio_test():
        # Compute reconstruction error when dropping random trios of components
        trio_drop_errors = {}
        component_trios = list(combinations(range(n_features), 3))

        for trio in component_trios:
            transformed_dropped = transformed_full.copy()
            transformed_dropped[:, trio[0]] = 0
            transformed_dropped[:, trio[1]] = 0
            transformed_dropped[:, trio[2]] = 0
            reconstructed_dropped = pca_full.inverse_transform(transformed_dropped)
            error = np.mean((data - reconstructed_dropped) ** 2)
            trio_drop_errors[trio] = error

        # Find the best trio to drop
        best_trio = min(trio_drop_errors, key=trio_drop_errors.get)
        print(f"Best trio to drop: {best_trio} with error: {trio_drop_errors[best_trio]}")

        # Compare with dropping the last three components
        last_three = (n_features-3, n_features-2, n_features-1)
        transformed_dropped = transformed_full.copy()
        transformed_dropped[:, last_three[0]] = 0
        transformed_dropped[:, last_three[1]] = 0
        transformed_dropped[:, last_three[2]] = 0
        reconstructed_dropped = pca_full.inverse_transform(transformed_dropped)
        last_three_error = np.mean((data - reconstructed_dropped) ** 2)
        print(f"Error when dropping last three components: {last_three_error}")

        # Plot reconstruction error for each trio
        trios, errors = zip(*trio_drop_errors.items())
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(errors)), errors, tick_label=[str(t) for t in trios], color='skyblue')
        plt.xticks(rotation=90)
        plt.xlabel("Dropped PCA Component Trios")
        plt.ylabel("Reconstruction Error")
        plt.title("Reconstruction Error When Dropping Component Trios")
        plt.show()

    _trio_test()

    return


if __name__ == "__main__":
    app.run()
