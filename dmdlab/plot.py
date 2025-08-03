import matplotlib.pyplot as plt
import numpy as np

def plot_eigs(eigs, **kwargs):
    """
    Plot the provided eigenvalues (of the dynamics operator A).
    
    Args:
        eigs (:obj:`ndarray` of complex): 
        **kwargs: kwargs of matplotlib.pyplot.subplots

    Returns:
        (tuple): Tuple containing:
            fig: figure object
            ax: axes object
    """
    xlim = kwargs.pop('xlim', [-1.1, 1.1])
    ylim = kwargs.pop('xlim', [-1.1, 1.1])

    fig, ax = plt.subplots(1, **kwargs)
    ax.set_aspect('equal'), ax.set_xlim(xlim), ax.set_ylim(ylim)
    ax.scatter(eigs.real, eigs.imag)
    ax.add_artist(plt.Circle((0, 0), 1, color='k', linestyle='--', fill=False))
    return fig, ax
def plot_eigs_history(eigs_history, **kwargs):
    """
    Plots the convergence of eigenvalues over time.

    Args:
        eigs_history (list of ndarray): A list where each element is an array
                                        of eigenvalues at a given time step.
    """
    fig, ax = plt.subplots(1, 1, **kwargs)
    
    # Transpose the history so we can plot each eigenvalue's trajectory
    # Use np.array to handle lists of arrays gracefully
    eigs_over_time = np.array(eigs_history).T

    # Plot the path of each eigenvalue
    for eig_path in eigs_over_time:
        # Filter out nans that may have occurred early in learning
        valid_eigs = eig_path[~np.isnan(eig_path)]
        ax.plot(valid_eigs.real, valid_eigs.imag, '.-', alpha=0.5, markersize=4)

    # Plot final eigenvalues more prominently
    final_eigs = eigs_over_time[:, -1]
    ax.scatter(final_eigs.real, final_eigs.imag, c='red', s=100, zorder=5, label='Final Eigenvalues')
    
    # Plot initial eigenvalues
    initial_eigs = eigs_over_time[:, 0]
    ax.scatter(initial_eigs.real, initial_eigs.imag, c='green', s=100, zorder=5, label='Initial Eigenvalues')


    # Add unit circle for reference
    ax.add_artist(plt.Circle((0, 0), 1, color='k', linestyle='--', fill=False))
    
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title("Eigenvalue Convergence Trajectory")
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    
    return fig, ax


# TODO: def hinton(args):
