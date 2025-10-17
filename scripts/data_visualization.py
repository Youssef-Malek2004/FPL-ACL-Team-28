import matplotlib.pyplot as plt

def plot_learning_curves(model) -> None:
    """
    Plots RMSE for learn/validation across iterations from CatBoost.
    """
    evals = model.get_evals_result()
    # Keys typically: 'learn' and 'validation' (or 'validation_0' if multiple)
    learn_rmse = evals.get('learn', {}).get('RMSE', None)
    # CatBoost may store validation under 'validation' or 'validation_0'
    valid_rmse = None
    if 'validation' in evals:
        valid_rmse = evals['validation'].get('RMSE', None)
    elif 'validation_0' in evals:
        valid_rmse = evals['validation_0'].get('RMSE', None)

    if learn_rmse is None or valid_rmse is None:
        print("No eval results found to plot. Ensure you passed eval_set in fit().")
        return

    iters = range(1, len(learn_rmse) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(iters, learn_rmse, label="Train RMSE")
    plt.plot(iters, valid_rmse, label="Valid RMSE")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("CatBoost Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()