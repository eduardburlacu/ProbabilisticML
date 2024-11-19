import scipy.linalg
import numpy as np
from tqdm import tqdm


def gibbs_sample(game_mat: np.ndarray, num_players: int, num_iter: int):
    """

    :param game_mat: The game matrix of shape [num_games, 2]. The first column is the id of the player who won,
        the second column is the id of the player who lost.
    :param num_players: Number of players
    :param num_iter: Number of iterations of Gibbs sampling to perform
    :return:
    """
    # number of games
    n_games = game_mat.shape[0]
    # Array containing mean skills of each player, set to prior mean
    w = np.zeros((num_players, 1))
    # Array that will contain skill samples
    skill_samples = np.zeros((num_players, num_iter))
    # Array containing skill variance for each player, set to prior variance
    player_var = 0.5 * np.ones(num_players)
    # number of iterations of Gibbs
    for i in tqdm(range(num_iter)):

        # sample performance given differences in skills and outcomes (using rejection sampling)
        t = np.zeros((n_games, 1))
        for g in range(n_games):
            s = w[game_mat[g, 0]] - w[game_mat[g, 1]]  # skill difference
            t[g] = s + np.random.randn()  # Sample performance
            while t[g] < 0:  # rejection step
                t[g] = s + np.random.randn()  # resample if rejected

        # Jointly sample skills given performance differences
        m = np.zeros((num_players, 1))  # the intermediate skill mean
        for p in range(num_players):
            m[p] = (((game_mat[:, 0] == p).astype(np.float32) - (game_mat[:, 1] == p).astype(np.float32)) * t[:, 0]).sum()
            # Note that prior mean is zero -> prior term not needed
        iS = np.zeros((num_players, num_players))  # Container for sum of precision matrices (likelihood terms)

        for g in range(n_games):
            # TODO: Build the iS matrix
            iS[game_mat[g, 0], game_mat[g, 0]] += 1
            iS[game_mat[g, 1], game_mat[g, 1]] += 1
            iS[game_mat[g, 0], game_mat[g, 1]] -= 1
            iS[game_mat[g, 1], game_mat[g, 0]] -= 1

        # Posterior precision matrix
        iSS = iS + np.diag(1. / player_var)

        # Use Cholesky decomposition to sample from a multivariate Gaussian
        iR = scipy.linalg.cho_factor(iSS)  # Cholesky decomposition of the posterior precision matrix
        mu = scipy.linalg.cho_solve(iR, m, check_finite=False)  # uses cholesky factor to compute inv(iSS) @ m

        # sample from N(mu, inv(iSS))
        w = mu + scipy.linalg.solve_triangular(iR[0], np.random.randn(num_players, 1), check_finite=False)
        skill_samples[:, i] = w[:, 0]
    return skill_samples
