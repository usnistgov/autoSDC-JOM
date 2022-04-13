import gpflow
import dataset
import numpy as np
import pandas as pd
import tensorflow as tf


def simplex_grid(n=3, buffer=0.1):
    """ construct a regular grid on the ternary simplex """

    xx, yy = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n))
    s = np.c_[xx.flat, yy.flat]

    sel = np.abs(s).sum(axis=1) <= 1.0
    s = s[sel]
    ss = 1 - s.sum(axis=1)
    s = np.hstack((s, ss[:, None]))

    scale = 1 - (3 * buffer)
    s = buffer + s * scale
    return s


def model_ternary(
    composition,
    target,
    reset_tf_graph=True,
    drop_last=True,
    optimize_noise_variance=True,
    initial_noise_var=1e-4,
):

    if drop_last:
        X = composition[:, :-1]  # ignore the last composition column
    else:
        X = composition
    Y = target

    # sel = np.isfinite(Y).sum(axis=1)
    sel = np.isfinite(Y).flat
    X, Y = X[sel], Y[sel]
    N, D = X.shape

    if reset_tf_graph:
        gpflow.reset_default_graph_and_session()

    with gpflow.defer_build():
        m = gpflow.models.GPR(
            X,
            Y,
            # kern=gpflow.kernels.Linear(D, ARD=True) + gpflow.kernels.RBF(D, ARD=True) + gpflow.kernels.Constant(D) + gpflow.kernels.White(D)
            kern=gpflow.kernels.Matern52(D, ARD=True)
            + gpflow.kernels.Constant(D)
            + gpflow.kernels.White(D, variance=initial_noise_var)  # \sigma_noise = 0.01
            # kern=gpflow.kernels.RationalQuadratic(D, ARD=True) + gpflow.kernels.Constant(D) + gpflow.kernels.White(D, variance=initial_noise_var)
        )

    # set a weakly-informative lengthscale prior
    # e.g. half-normal(0, dx/3) -> gamma(0.5, 2*dx/3)
    # another choice might be to use an inverse gamma prior...
    # m.kern.kernels[0].lengthscales.prior = gpflow.priors.Gamma(0.5, 2.0/3)
    m.kern.kernels[0].lengthscales.prior = gpflow.priors.Gamma(0.5, 0.5)

    # m.kern.kernels[0].variance.prior = gpflow.priors.Gamma(0.5, 4.)
    # m.kern.kernels[1].variance.prior = gpflow.priors.Gamma(0.5, 4.)
    # m.kern.kernels[2].variance.prior = gpflow.priors.Gamma(0.5, 2.)
    m.kern.kernels[0].variance.prior = gpflow.priors.Gamma(2.0, 2.0)
    m.kern.kernels[1].variance.prior = gpflow.priors.Gamma(2.0, 2.0)
    # m.kern.kernels[2].variance.prior = gpflow.priors.Gamma(2.0, 2.0)

    if not optimize_noise_variance:
        m.kern.kernels[2].variance.trainable = False

    m.likelihood.variance = 1e-6

    m.compile()
    return m


def model_property(X, y, dx=1.0, optimize=False):

    sel = np.isfinite(y).flat
    X, y = X[sel], y[sel]
    N, D = X.shape

    with gpflow.defer_build():
        model = gpflow.models.GPR(
            X, y, kern=gpflow.kernels.RBF(D, ARD=True) + gpflow.kernels.Constant(D),
        )

        model.kern.kernels[0].variance.prior = gpflow.priors.Gamma(2, 1 / 2)
        model.kern.kernels[0].lengthscales.prior = gpflow.priors.Gamma(2.0, 2 * dx / 3)
        model.likelihood.variance = 0.01

    model.compile()

    if optimize:
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(model)

    return model


def model_quality(X, y, dx=1.0, likelihood="beta", optimize=False):

    sel = np.isfinite(y).flat
    X, y = X[sel], y[sel]
    N, D = X.shape

    if likelihood == "beta":
        # bounded regression
        lik = gpflow.likelihoods.Beta()
    elif likelihood == "bernoulli":
        # classification
        lik = gpflow.likelihoods.Bernoulli()

    with gpflow.defer_build():
        model = gpflow.models.VGP(
            X, y, kern=gpflow.kernels.RBF(D, ARD=True), likelihood=lik
        )

        model.kern.variance.prior = gpflow.priors.Gamma(2, 2)
        model.kern.lengthscales.prior = gpflow.priors.Gamma(1.0, 2 * dx / 3)
        model.likelihood.variance = 0.1

    model.compile()

    if optimize:
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(model)

    return model


class NiTiAlEmulator:
    def __init__(
        self,
        composition,
        df,
        components=["Ni", "Al", "Ti"],
        targets=["V_oc", "I_p", "V_tp", "slope", "fwhm"],
        dx=1.0,
    ):
        """ fit independent GP models for each target -- read compositions and targets from a csv file... """

        self.composition = composition
        self.components = list(composition.columns)
        self.df = df
        self.targets = targets
        # self.composition = self.df.loc[:,self.components].values
        self.dx = dx

        self.models = {}
        self.session = gpflow.get_default_session()
        self.opt = gpflow.training.ScipyOptimizer()

        self.fit()

    def fit(self):
        print(self.composition.shape)
        with self.session.as_default():
            for target in self.targets:

                model = model_property(
                    self.composition.values[:, :-1],
                    self.df[target].values[:, None],
                    dx=self.dx,
                )
                self.opt.minimize(model)
                self.models[target] = model

    def likelihood_variance(self, target=None):
        return self.models[target].likelihood.variance.value.item()

    def __call__(
        self,
        composition,
        target=None,
        return_var=False,
        sample_posterior=False,
        n_samples=1,
        seed=None,
    ):
        """ evaluate GP models on compositions """
        model = self.models[target]

        with self.session.as_default():
            if sample_posterior:
                if seed is not None:
                    tf.set_random_seed(seed)
                mu = model.predict_f_samples(composition[:, :-1], n_samples)
                return mu.squeeze()
            else:
                mu, var = model.predict_y(composition[:, :-1])
                if return_var:
                    return mu, var
                else:
                    return mu.squeeze()


class ExperimentEmulator:
    def __init__(
        self,
        db_file,
        components=["Ni", "Al", "Ti"],
        targets=["V_oc", "I_p", "V_tp", "slope", "fwhm"],
        optimize_noise_variance=True,
    ):
        """ fit independent GP models for each target -- read compositions and targets from a csv file... """

        # load all the unflagged data from sqlite to pandas
        # use sqlite id as pandas index
        self.db = dataset.connect(f"sqlite:///{db_file}")
        self.df = pd.DataFrame(self.db["experiment"].all(flag=False))
        self.df.set_index("id", inplace=True)

        # # drop the anomalous point 45 that has a negative jog in the passivation...
        # self.df = self.df.drop(45)

        self.components = components
        self.targets = targets
        self.optimize_noise_variance = optimize_noise_variance

        self.models = {}
        self.fit()

    def fit(self):
        self.composition = self.df.loc[:, self.components].values

        self.opt = gpflow.training.ScipyOptimizer()
        for target in self.targets:
            model = model_ternary(
                self.composition,
                self.df[target].values[:, None],
                optimize_noise_variance=self.optimize_noise_variance,
            )
            session = gpflow.get_default_session()
            self.opt.minimize(model)
            self.models[target] = (session, model)

    def __call__(
        self,
        composition,
        target=None,
        return_var=False,
        sample_posterior=False,
        n_samples=1,
        seed=None,
    ):
        """ evaluate GP models on compositions """
        session, model = self.models[target]

        with session.as_default():
            if sample_posterior:
                if seed is not None:
                    tf.set_random_seed(seed)
                mu = model.predict_f_samples(composition[:, :-1], n_samples)
                return mu.squeeze()
            else:
                mu, var = model.predict_y(composition[:, :-1])
                if return_var:
                    return mu, var
                else:
                    return mu.squeeze()
