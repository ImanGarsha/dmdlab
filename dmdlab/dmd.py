# DMD algorithms by Andy Goldschmidt.
#
# TODO:
# - Should we create an ABC interface for DMD?
# - __init__.py and separate files
#
import numpy as np
from numpy.linalg import svd, pinv, eig
from scipy.linalg import expm

from .process import _threshold_svd, dag


class DMD:
    def __init__(self, X2, X1, ts, **kwargs):
        """ X2 = A X1

        Args:
            X2 (:obj:`ndarray` of float): Left side data matrix with columns containing states at sequential times.
            X1 (:obj:`ndarray` of float): Right side data matrix with columns containing states at sequential times.
            U (:obj:`ndarray` of float): Control signal(s) with columns containing controls.
            ts (:obj:`ndarray` of float): Time measurements
            **kwargs: see Keyword arguments.

        Keyword arguments:
            threshold (real, int): Truncate the singular values associated with DMD modes. default None.
            threshold_type (str): One of {'number', 'percent'}. default 'percent'.

        Attributes:
            X2 (:obj:`ndarray` of float): Left side data matrix
            X1 (:obj:`ndarray` of float): Right side data matrix
            U (:obj:`ndarray` of float): Control signal data matrix
            t0 (float): Initial time.
            dt (float): Step size.
            orig_timesteps (:obj:`ndarray` of float): Original times matching X1.
            A (:obj:`ndarray` of float): Learned drift operator.
            Atilde (:obj:`ndarray` of float): Projected A.
            eigs (list of float): Eigenvalues of Atilde.
            modes (:obj:`ndarray` of float): DMD modes are eigenvectors of Atilde (shared by A).
        """
        self.X2 = X2
        self.X1 = X1

        self.t0 = ts[0]
        self.dt = ts[1] - ts[0]
        self.orig_timesteps = ts if len(ts) == self.X1.shape[1] else ts[:-1]

        # I. Compute SVD
        threshold = kwargs.get('threshold', None)
        if threshold is None:
            U, S, Vt = svd(self.X1, full_matrices=False)
        else:
            threshold_type = kwargs.get('threshold_type', 'percent')
            U, S, Vt = _threshold_svd(self.X1, threshold, threshold_type)

        # II: Compute operators: X2 = A X1 and Atilde = U*AU
        Atilde = dag(U) @ self.X2 @ dag(Vt) @ np.diag(1 / S)
        self.A = self.X2 @ dag(Vt) @ np.diag(1 / S) @ dag(U)

        # III. DMD Modes
        #       Atilde W = W Y (Eigendecomposition)
        self.eigs, W = eig(Atilde)

        # Two versions (eigenvectors of A)
        #       (i)  DMD_exact = X2 V S^-1 W
        #       (ii) DMD_proj = U W
        dmd_modes = kwargs.get('dmd_modes', 'exact')
        if dmd_modes == 'exact':
            self.modes = self.X2 @ dag(Vt) @ np.diag(1 / S) @ W
        elif dmd_modes == 'projected':
            self.modes = U @ W
        else:
            raise ValueError('In DMD initialization, unknown dmd_mode type.')

    @classmethod
    def from_full(cls, X, ts, **kwargs):
        X1 = X[:, :-1]
        X2 = X[:, 1:]
        return cls(X2, X1, ts, **kwargs)

    def time_spectrum(self, ts, system='discrete'):
        """Returns a continuous approximation to the time dynamics of A.

        Note that A_dst = e^(A_cts dt). Suppose (operator, eigs) pairs are denoted (A_dst, Y) for the discrete case
        and (A_cts, Omega) for the continuous case. The eigenvalue correspondence is e^log(Y)/dt = Omega.

        Args:
            ts (:obj:`ndarray` of float): Times.
            system ({'continuous', 'discrete'}): default 'discrete'.

        Returns:
            :obj:`ndarray` of float: Evaluations of modes at ts.
        """
        if np.isscalar(ts):
            # Cast eigs to complex numbers for logarithm
            if system == 'discrete':
                omega = np.log(self.eigs + 0j) / self.dt
            elif system == 'continuous':
                omega = self.eigs + 0j
            else:
                raise ValueError('In time_spectrum, invalid system value.')
            return np.exp(omega * (ts - self.t0))
        else:
            return np.array([self.time_spectrum(it, system=system) for it in ts]).T

    def _predict(self, ts, x0, system):
        left = self.modes
        right = pinv(self.modes) @ x0
        if np.isscalar(ts):
            return left @ np.diag(self.time_spectrum(ts, system)) @ right
        else:
            return np.array([left @ np.diag(self.time_spectrum(it, system)) @ right for it in ts]).T

    def predict_dst(self, ts=None, x0=None):
        """Predict the future state using continuous approximation to the discrete A.

        Args:
            ts (:obj:`ndarray` of float): Array of time-steps to predict. default self.orig_timesteps.
            x0 (:obj:`ndarray` of float): The initial value. default self.x0.

        Returns:
            :obj:`ndarray` of float: Predicted state for each control input.
        """
        x0 = self.X1[:, 0] if x0 is None else x0
        ts = self.orig_timesteps if ts is None else ts
        return self._predict(ts, x0, 'discrete')

    def predict_cts(self, ts=None, x0=None):
        """Predict the future state using the continuous operator A.

        Args:
            ts (:obj:`ndarray` of float): Array of time-steps to predict. default self.orig_timesteps.
            x0 (:obj:`ndarray` of float): The initial value. default self.x0.

        Returns:
             :obj:`ndarray` of float: Predicted state for each control input.
        """
        x0 = self.X1[:, 0] if x0 is None else x0
        ts = self.orig_timesteps if ts is None else ts
        return self._predict(ts, x0, 'continuous')


class DMDc:
    def __init__(self, X2, X1, U, ts, **kwargs):
        """ X2 = A X1 + B U

        Args:
            X2 (:obj:`ndarray` of float): Left side data matrix with columns containing states at sequential times.
            X1 (:obj:`ndarray` of float): Right side data matrix with columns containing states at sequential times.
            U (:obj:`ndarray` of float): Control signal(s) with columns containing controls.
            ts (:obj:`ndarray` of float): Time measurements
            **kwargs: see Keyword arguments.

        Keyword arguments:
            threshold (real, int): Truncate the singular values associated with DMD modes. default None.
            threshold_type (str): One of {'number', 'percent'}. default 'percent'.

        Attributes:
            X2 (:obj:`ndarray` of float): Left side data matrix
            X1 (:obj:`ndarray` of float): Right side data matrix
            U (:obj:`ndarray` of float): Control signal data matrix
            t0 (float): Initial time.
            dt (float): Step size.
            orig_timesteps (:obj:`ndarray` of float): Original times matching X1.
            A (:obj:`ndarray` of float): Learned drift operator.
            Atilde (:obj:`ndarray` of float): Projected A.
            B (:obj:`ndarray` of float): Learned control operator.
            Btilde (:obj:`ndarray` of float): projected B.
            eigs (list of float): Eigenvalues of Atilde.
            modes (:obj:`ndarray` of float): DMD modes are eigenvectors of Atilde (shared by A).
        """
        self.X1 = X1
        self.X2 = X2
        self.U = U if U.shape[1] == self.X1.shape[1] else U[:, :-1]  # ONLY these 2 options
        Omega = np.vstack([self.X1, self.U])

        self.t0 = ts[0]
        self.dt = ts[1] - ts[0]
        self.orig_timesteps = ts if len(ts) == self.X1.shape[1] else ts[:-1]

        # I. Compute SVDs
        threshold = kwargs.get('threshold', None)
        if threshold is None:
            Ug, Sg, Vgt = svd(Omega, full_matrices=False)
            U, S, Vt = svd(self.X2, full_matrices=False)
        else:
            # Allow for independent thresholding
            t1, t2 = 2 * [threshold] if np.isscalar(threshold) else threshold
            threshold_type = kwargs.get('threshold_type', 'percent')
            Ug, Sg, Vgt = _threshold_svd(Omega, t1, threshold_type)
            U, S, Vt = _threshold_svd(self.X2, t2, threshold_type)

        # II. Compute operators
        n, _ = self.X2.shape
        left = self.X2 @ dag(Vgt) @ np.diag(1 / Sg)
        self.A = left @ dag(Ug[:n, :])
        self.B = left @ dag(Ug[n:, :])

        # III. DMD modes
        self.Atilde = dag(U) @ self.A @ U
        self.Btilde = dag(U) @ self.B
        self.eigs, W = eig(self.Atilde)
        self.modes = self.A @ U @ W

    @classmethod
    def from_full(cls, X, U, ts, **kwargs):
        X2 = X[:, 1:]
        X1 = X[:, :-1]
        return cls(X2, X1, U, ts, **kwargs)

    def predict_dst(self, control=None, x0=None):
        """ Predict the future state using discrete evolution.

        Evolve the system from X0 as long as control is available, using
        the discrete evolution X_2 = A X_1 + B u_1.

        Default behavior (control=None) is to use the original control. (If the underlying A is desired,
        format zeros_like u that runs for the desired time.)

        Args:
            control (:obj:`ndarray` of float): The control signal.
            x0 (:obj:`ndarray` of float): The initial value.

        Returns:
            :obj:`ndarray` of float: Predicted state for each control input.
        """
        U = self.U if control is None else control
        xt = self.X1[:, 0] if x0 is None else x0
        res = [xt]
        for ut in U[:, :-1].T:
            xt_1 = self.A @ xt + self.B @ ut
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def predict_cts(self, control=None, x0=None, dt=None):
        """ Predict the future state using continuous evolution.

        Evolve the system from X0 as long as control is available, using
        the continuous evolution while u is constant,

            X_dot = A X + B u
            x(t+dt) = e^{dt A}(x(t) + dt B u(t))

        Default behavior (control=None) is to use the original control. (If the underlying A is desired,
        format zeros_like u that runs for the desired time.) Be sure that dt matches the train dt if
        using delay embeddings.

        Args:
            control (:obj:`ndarray` of float): The control signal.
                A zero-order hold is assumed between time steps.
                The dt must match the training data if time-delays are used.
            x0 (:obj:`ndarray` of float): The initial value.
            dt (float): The time-step between control inputs.

        Returns:
            :obj:`ndarray` of float: Predicted state for each control input.
        """
        U = self.U if control is None else control
        dt = self.dt if dt is None else dt
        xt = self.X1[:, 0] if x0 is None else x0
        res = [xt]
        for ut in U[:, :-1].T:
            xt_1 = expm(dt * self.A) @ (xt + dt * self.B @ ut)
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def zero_control(self, n_steps=None):
        n_steps = len(self.orig_timesteps) if n_steps is None else n_steps
        return np.zeros([self.U.shape[0], n_steps])


class biDMD:
    def __init__(self, X2, X1, U, ts, **kwargs):
        """X2 = A X1 + U B X1

        Args:
            X2 (:obj:`ndarray` of float): Left side data matrix with columns containing states at sequential times.
            X1 (:obj:`ndarray` of float): Right side data matrix with columns containing states at sequential times.
            U (:obj:`ndarray` of float): Control signal(s) with columns containing controls.
            ts (:obj:`ndarray` of float): Time measurements
            **kwargs: see Keyword arguments.

        Keyword arguments:
            shift (int): Number of time delays in order to match times in the nonlinear term. default 0.
            threshold (real, int): Truncate the singular values associated with DMD modes. default None.
            threshold_type (str): One of {'number', 'percent'}. default 'percent'.

        Attributes:
            X2 (:obj:`ndarray` of float): Left side data matrix
            X1 (:obj:`ndarray` of float): Right side data matrix
            U (:obj:`ndarray` of float): Control signal data matrix
            Ups (:obj:`ndarray` of float): augmented state U*X1.
            t0 (float): Initial time.
            dt (float): Step size.
            orig_timesteps (:obj:`ndarray` of float): Original times matching X1.
            A (:obj:`ndarray` of float): Learned drift operator.
            Atilde (:obj:`ndarray` of float): Projected A.
            B (:obj:`ndarray` of float): Learned nonlinear control operator.
            Btilde (:obj:`ndarray` of float): projected B.
            eigs (list of float): Eigenvalues of Atilde.
            modes (:obj:`ndarray` of float): DMD modes are eigenvectors of Atilde (shared by A).
        """
        self.U = U
        self.X1 = X1
        self.X2 = X2

        self.t0 = ts[0]
        self.dt = ts[1] - ts[0]
        self.orig_timesteps = ts if len(ts) == self.X1.shape[1] else ts[:-1]

        # store useful dimension
        n_time = len(self.orig_timesteps)

        # Partially unwrap delay embedding to make sure the correct control signals
        #   are combined with the correct data times. The unwrapped (=>) operators:
        #     X1  => (delays+1) x (measured dimensions) x (measurement times)
        #     U   => (delays+1) x (number of controls)  x (measurement times)
        #     Ups => (delays+1) x (controls) x (measured dimensions) x (measurement times)
        #         => (delays+1 x controls x measured dimensions) x (measurement times)
        #   Re-flatten all but the time dimension of Ups to set the structure of the
        #   data matrix. This will set the strucutre of the B operator to match our
        #   time-delay function.
        self.shift = kwargs.get('shift', 0)
        self.Ups = np.einsum('sct, smt->scmt',
                             self.U.reshape(self.shift + 1, -1, n_time),
                             self.X1.reshape(self.shift + 1, -1, n_time)
                             ).reshape(-1, n_time)
        Omega = np.vstack([self.X1, self.Ups])

        # I. Compute SVDs
        threshold = kwargs.get('threshold', None)
        if threshold is None:
            Ug, Sg, Vgt = svd(Omega, full_matrices=False)
            U, S, Vt = svd(self.X2, full_matrices=False)
        else:
            # Allow for independent thresholding
            t1, t2 = 2 * [threshold] if np.isscalar(threshold) else threshold
            threshold_type = kwargs.get('threshold_type', 'percent')
            Ug, Sg, Vgt = _threshold_svd(Omega, t1, threshold_type)
            U, S, Vt = _threshold_svd(self.X2, t2, threshold_type)

        # II. Compute operators
        n, _ = self.X2.shape
        left = self.X2 @ dag(Vgt) @ np.diag(1 / Sg)
        self.A = left @ dag(Ug[:n, :])
        self.B = left @ dag(Ug[n:, :])

        # III. DMD modes
        self.Atilde = dag(U) @ self.A @ U
        self.Btilde = dag(U) @ self.B
        self.eigs, W = eig(self.Atilde)
        self.modes = self.A @ U @ W

    def predict_dst(self, control=None, x0=None):
        """ Predict the future state using discrete evolution.

        Evolve the system from X0 as long as control is available, using
        the discrete evolution:

            x_1 = A x_0 + B (u.x_0)
                = [A B] [x_0, u.x_0]^T

        Args:
            control (:obj:`ndarray` of float): The control signal.
            x0 (): The initial value.

        Returns:
            :obj:`ndarray` of float: Predicted state for each control input.
        """
        control = self.U if control is None else control
        xt = self.X1[:, 0] if x0 is None else x0  # Flat array
        res = [xt]
        for t in range(control.shape[1] - 1):
            # Outer product then flatten to correctly combine the different
            #   times present due to time-delays. That is, make sure that
            #   u(t)'s multiply x(t)'s
            #     _ct    => (time-delays + 1) x (number of controls)
            #     _xt    => (time-delays + 1) x (measured dimensions)
            #     _ups_t => (time-delays + 1) x (controls) x (measurements)
            #   Flatten to get the desired vector.
            _ct = control[:, t].reshape(self.shift + 1, -1)
            _xt = xt.reshape(self.shift + 1, -1)
            ups_t = np.einsum('sc,sm->scm', _ct, _xt).flatten()

            xt_1 = self.A @ xt + self.B @ ups_t
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def predict_cts(self, control=None, x0=None, dt=None):
        """ Predict the future state using continuous evolution.

        Evolve the system from X0 as long as control is available, using
        the continuous evolution while u is constant,

            x_{t+1} = e^{A dt + u B dt } x_t

        Args:
            control (:obj:`ndarray` of float): The control signal.
                A zero-order hold is assumed between time steps.
                The dt must match the training data if time-delays are used.
            x0 (:obj:`ndarray` of float): The initial value.
            dt (float): The time-step between control inputs.

        Returns:
            :obj:`ndarray` of float: Predicted state for each control input.
        """
        control = self.U if control is None else control
        dt = self.dt if dt is None else dt
        xt = self.X1[:, 0] if x0 is None else x0  # Flat array

        # store useful dimensions
        delay_dim = self.shift + 1
        control_dim = self.U.shape[0] // delay_dim
        measure_1_dim = self.X1.shape[0] // delay_dim
        to_dim = self.X2.shape[0]

        res = [xt]
        for t in range(control.shape[1] - 1):
            # Correctly combine u(t) and B(t)
            #   Initial:
            #     B      <= (time-delays+1 x measurements_2) x (time-delays+1 x controls x measurements_1)
            #   Reshape:
            #     B      => (time-delays+1 x measurements_2) x (time-delays+1) x (controls) x (measurements_1)
            #     _ct    => (time-delays+1) x (controls)
            #     _uBt   => (time-delays+1 x measurements_2) x (time-delays+1) x (measurements_1)
            #            => (time-delays+1 x measurements_2) x (time-delays+1 x measurements_1)
            #   Notice that _uBt is formed by a sum over all controls in order to act on the
            #   state xt which has dimensions of (delays x measurements_1).
            _uBt = np.einsum('ascm,sc->asm',
                             self.B.reshape(to_dim, delay_dim, control_dim, measure_1_dim),
                             control[:, t].reshape(delay_dim, control_dim)
                             ).reshape(to_dim, delay_dim * measure_1_dim)

            xt_1 = expm((self.A + _uBt) * dt) @ xt
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def zero_control(self, n_steps=None):
        n_steps = len(self.orig_timesteps) if n_steps is None else n_steps
        return np.zeros([self.Ups.shape[0], n_steps])


class biDMDc:
    def __init__(self, X2, X1, U, ts, **kwargs):
        """ X2 = A X1 + U B X1 + D U

        Args:
            X2 (:obj:`ndarray` of float): Left side data matrix with columns containing states at sequential times.
            X1 (:obj:`ndarray` of float): Right side data matrix with columns containing states at sequential times.
            U (:obj:`ndarray` of float): Control signal(s) with columns containing controls.
            ts (:obj:`ndarray` of float): Time measurements
            **kwargs: see Keyword arguments.
            
        Keyword arguments:
            shift (int): Number of time delays in order to match times in the nonlinear term. default 0.
            threshold (real, int): Truncate the singular values associated with DMD modes. default None.
            threshold_type (str): One of {'number', 'percent'}. default 'percent'.

        Attributes:
            X2 (:obj:`ndarray` of float): Left side data matrix
            X1 (:obj:`ndarray` of float): Right side data matrix
            U (:obj:`ndarray` of float): Control signal data matrix
            Ups (:obj:`ndarray` of float): augmented state U*X1.
            t0 (float): Initial time.
            dt (float): Step size.
            orig_timesteps (:obj:`ndarray` of float): Original times matching X1.
            A (:obj:`ndarray` of float): Learned drift operator.
            Atilde (:obj:`ndarray` of float): Projected A.
            B (:obj:`ndarray` of float): Learned nonlinear control operator.
            Btilde (:obj:`ndarray` of float): projected B.
            D (:obj:`ndarray` of float): Learned control operator.
            eigs (list of float): Eigenvalues of Atilde.
            modes (:obj:`ndarray` of float): DMD modes are eigenvectors of Atilde (shared by A).
        """
        self.U = U
        self.X1 = X1
        self.X2 = X2

        self.t0 = ts[0]
        self.dt = ts[1] - ts[0]
        self.orig_timesteps = ts if len(ts) == self.X1.shape[1] else ts[:-1]

        # store useful dimension
        n_time = len(self.orig_timesteps)
        self.shift = kwargs.get('shift', 0)
        delay_dim = self.shift + 1

        # Partially unwrap delay embedding to make sure the correct control signals
        #   are combined with the correct data times. The unwrapped (=>) operators:
        #     X1  => (delays+1) x (measured dimensions) x (measurement times)
        #     U   => (delays+1) x (number of controls)  x (measurement times)
        #     Ups => (delays+1) x (controls) x (measured dimensions) x (measurement times)
        #         => (delays+1 x controls x measured dimensions) x (measurement times)
        #   Re-flatten all but the time dimension of Ups to set the structure of the
        #   data matrix. This will set the structure of the B operator to match our
        #   time-delay function.
        self.Ups = np.einsum('sct, smt->scmt',
                             self.U.reshape(delay_dim, -1, n_time),
                             self.X1.reshape(delay_dim, -1, n_time)
                             ).reshape(-1, n_time)
        Omega = np.vstack([self.X1, self.Ups, self.U])

        # I. Compute SVDs
        threshold = kwargs.get('threshold', None)
        if threshold is None:
            Ug, Sg, Vgt = svd(Omega, full_matrices=False)
            U, S, Vt = svd(self.X2, full_matrices=False)
        else:
            # Allow for independent thresholding
            t1, t2 = 2 * [threshold] if np.isscalar(threshold) else threshold
            threshold_type = kwargs.get('threshold_type', 'percent')
            Ug, Sg, Vgt = _threshold_svd(Omega, t1, threshold_type)
            U, S, Vt = _threshold_svd(self.X2, t2, threshold_type)

        # II. Compute operators
        c = self.U.shape[0] // delay_dim
        n = self.X1.shape[0]
        left = self.X2 @ dag(Vgt) @ np.diag(1 / Sg)
        # Omega = X + uX + u => dim'ns: n + c*n + c
        self.A = left @ dag(Ug[:n, :])
        self.B = left @ dag(Ug[n:(c + 1) * n, :])
        self.D = left @ dag(Ug[(c + 1) * n:, :])

        # III. DMD modes
        self.Atilde = dag(U) @ self.A @ U
        self.Btilde = dag(U) @ self.B
        self.Dtilde = dag(U) @ self.D
        self.eigs, W = eig(self.Atilde)
        self.modes = self.A @ U @ W

    def predict_dst(self, control=None, x0=None):
        """ Predict the future state using discrete evolution.

        Evolve the system from X0 as long as control is available, using
        the discrete evolution,

            x_1 = A x_0 + B (u*x_0) + D u
                = [A B D] [x_0, u*x_0, u ]^T
        
        Args:
            control (:obj:`ndarray` of float): The control signal.
            x0 (): The initial value.

        Returns:
            :obj:`ndarray` of float: Predicted state for each control input.
        """
        control = self.U if control is None else control
        xt = self.X1[:, 0] if x0 is None else x0  # Flat array
        res = [xt]
        for t in range(control.shape[1] - 1):
            # Outer product then flatten to correctly combine the different
            #   times present due to time-delays. That is, make sure that
            #   u(t)'s multiply x(t)'s
            #     _ct    => (time-delays + 1) x (number of controls)
            #     _xt    => (time-delays + 1) x (measured dimensions)
            #     _ups_t => (time-delays + 1) x (controls) x (measurements)
            #   Flatten to get the desired vector.
            _ct = control[:, t].reshape(self.shift + 1, -1)
            _xt = xt.reshape(self.shift + 1, -1)
            ups_t = np.einsum('sc,sm->scm', _ct, _xt).flatten()

            xt_1 = self.A @ xt + self.B @ ups_t + self.D @ control[:, t]
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def predict_cts(self, control=None, x0=None, dt=None):
        """ Predict the future state using continuous evolution.

        Evolve the system from X0 as long as control is available, using
        the continuous evolution while u is constant,

            x_{t+1} = e^{A dt + u B dt } (x_t + dt * D u_t}

        Args:
            control (:obj:`ndarray` of float): The control signal.
                A zero-order hold is assumed between time steps.
                The dt must match the training data if time-delays are used.
            x0 (:obj:`ndarray` of float): The initial value.
            dt (float): The time-step between control inputs.

        Returns:
            :obj:`ndarray` of float: Predicted state for each control input.
        """
        control = self.U if control is None else control
        dt = self.dt if dt is None else dt
        xt = self.X1[:, 0] if x0 is None else x0  # Flat array

        # store useful dimensions
        delay_dim = self.shift + 1
        control_dim = self.U.shape[0] // delay_dim
        measure_1_dim = self.X1.shape[0] // delay_dim
        to_dim = self.X2.shape[0]

        res = [xt]
        for t in range(control.shape[1] - 1):
            # Correctly combine u(t) and B(t)
            #   Initial:
            #     B      <= (time-delays+1 x measurements_2) x (time-delays+1 x controls x measurements_1)
            #   Reshape:
            #     B      => (time-delays+1 x measurements_2) x (time-delays+1) x (controls) x (measurements_1)
            #     _ct    => (time-delays+1) x (controls) 
            #     _uBt   => (time-delays+1 x measurements_2) x (time-delays+1) x (measurements_1)
            #            => (time-delays+1 x measurements_2) x (time-delays+1 x measurements_1)
            #   Notice that _uBt is formed by a sum over all controls in order to act on the
            #   state xt which has dimensions of (delays x measurements_1).
            _uBt = np.einsum('ascm,sc->asm',
                             self.B.reshape(to_dim, delay_dim, control_dim, measure_1_dim),
                             control[:, t].reshape(delay_dim, control_dim)
                             ).reshape(to_dim, delay_dim * measure_1_dim)

            xt_1 = expm(dt * (self.A + _uBt)) @ (xt + dt * self.D @ control[:, t])
            xt = xt_1
            res.append(xt_1)
        return np.array(res).T

    def zero_control(self, n_steps=None):
        n_steps = len(self.orig_timesteps) if n_steps is None else n_steps
        return np.zeros([self.Ups.shape[0], n_steps])



class ibiDMD(biDMD):
    """
    Incremental Bilinear Dynamic Mode Decomposition (ibiDMD) with optional
    forgetting factor for tracking time-varying systems.
    """
    def __init__(self, X2, X1, U, ts, **kwargs):
        """
        Initializes the ibiDMD object and performs incremental learning.

        Args:
            X2 (:obj:`ndarray` of float): Left side data matrix.
            X1 (:obj:`ndarray` of float): Right side data matrix.
            U (:obj:`ndarray` of float): Control signal(s) data matrix.
            ts (:obj:`ndarray` of float): Time measurements.
            **kwargs: See Keyword arguments.

        Keyword arguments:
            shift (int): Number of time delays. default 0.
            p_alpha (float): Initialization value for the covariance matrix P.
                             P is initialized to p_alpha * I. Default 1e6.
            forgetting_factor (float): Forgetting factor (lambda) for tracking
                                       drifting systems. Must be between 0 and 1.
                                       Default is 1.0 (standard RLS).
            track_history (bool): If True, stores error and eigenvalue history
                                  for analysis. This requires looking at the
                                  full X2 matrix upfront, making it not purely
                                  online. Defaults to False.
        """
        # --- Stage 1: Initialization ---
        self.U = U
        self.X1 = X1
        self.X2 = X2
        self.t0 = ts[0]
        self.dt = ts[1] - ts[0]
        self.orig_timesteps = ts if len(ts) == self.X1.shape[1] else ts[:-1]
        self.shift = kwargs.get('shift', 0)
        self.forgetting_factor = kwargs.get('forgetting_factor', 1.0)
        track_history = kwargs.get('track_history', False)

        if not (0 < self.forgetting_factor <= 1.0):
            raise ValueError("Forgetting factor must be in the interval (0, 1].")

        # Get dimensions
        n_x1, n_time = self.X1.shape
        n_x2 = self.X2.shape[0]
        n_u = self.U.shape[0]
        
        delay_dim = self.shift + 1
        n_ups = (n_u // delay_dim) * (n_x1 // delay_dim) * delay_dim**2
        dim_xi = n_x1 + n_ups
        
        G = np.zeros((n_x2, dim_xi))
        p_alpha = kwargs.get('p_alpha', 1e6)
        P = np.identity(dim_xi) * p_alpha
        
        # --- History Tracking Initialization (Optional) ---
        if track_history:
            self.error_history = []
            self.eigs_history = []
            threshold = kwargs.get('threshold', None)
            if threshold is None:
                U_svd, _, _ = svd(self.X2, full_matrices=False)
            else:
                threshold_type = kwargs.get('threshold_type', 'percent')
                U_svd, _, _ = _threshold_svd(self.X2, threshold, threshold_type)
        else:
            self.error_history = None
            self.eigs_history = None

        # --- Stage 2: Incremental Update Loop (RLS) ---
        for t in range(n_time):
            x1_t = self.X1[:, t]
            x2_t = self.X2[:, t]
            u_t = self.U[:, t]

            _ct = u_t.reshape(self.shift + 1, -1)
            _xt = x1_t.reshape(self.shift + 1, -1)
            ups_t = np.einsum('sc,sm->scm', _ct, _xt).flatten()
            xi_t = np.hstack([x1_t, ups_t])

            # RLS Update Equations with Forgetting Factor
            # 1. Calculate gain vector (K)
            denominator = self.forgetting_factor + xi_t.T @ P @ xi_t
            if np.isclose(denominator, 0):
                K = np.zeros_like(xi_t)
            else:
                K = (P @ xi_t) / denominator

            # 2. Calculate prediction error (epsilon)
            epsilon = x2_t - G @ xi_t
            if track_history:
                self.error_history.append(np.linalg.norm(epsilon))

            # 3. Update system matrix G
            G = G + np.outer(epsilon, K)

            # 4. Update inverse covariance matrix P
            P = (P - np.outer(K, xi_t.T @ P)) / self.forgetting_factor

            # --- Store Eigenvalues for this step (Optional) ---
            if track_history:
                try:
                    current_A = G[:, :n_x1]
                    current_Atilde = dag(U_svd) @ current_A @ U_svd
                    current_eigs, _ = eig(current_Atilde)
                    self.eigs_history.append(current_eigs)
                except np.linalg.LinAlgError:
                    self.eigs_history.append(np.full(U_svd.shape[1], np.nan, dtype=np.complex128))

        # --- Stage 3: Finalize Matrices and Modes ---
        self.A = G[:, :n_x1]
        self.B = G[:, n_x1:]
        
        if not track_history:
            threshold = kwargs.get('threshold', None)
            if threshold is None:
                U_svd, _, _ = svd(self.X2, full_matrices=False)
            else:
                threshold_type = kwargs.get('threshold_type', 'percent')
                U_svd, _, _ = _threshold_svd(self.X2, threshold, threshold_type)
        
        self.Atilde = dag(U_svd) @ self.A @ U_svd
        self.Btilde = dag(U_svd) @ self.B
        self.eigs, W = eig(self.Atilde)
        self.modes = self.A @ U_svd @ W
        
        self.Ups = np.einsum('sct, smt->scmt',
                             self.U.reshape(self.shift + 1, -1, n_time),
                             self.X1.reshape(self.shift + 1, -1, n_time)
                             ).reshape(-1, n_time)
# Paste this entire class at the end of dmdlab/dmd.py

class Bayesian_biDMD(biDMD):
    """
    Bayesian Incremental Bilinear Dynamic Mode Decomposition.
    Inherits from biDMD for data handling and consistency.
    """
    def __init__(self, *args, rank=None, forgetting_factor=1.0, **kwargs):
        # 1. Call the parent constructor robustly with all its expected arguments
        super().__init__(*args, rank=rank, **kwargs)
        
        # 2. Set Bayesian-specific parameters
        if not (0 < forgetting_factor <= 1.0):
            raise ValueError("Forgetting factor must be in (0, 1].")
        self.lam = forgetting_factor
        
        self.M_G = None   # Posterior Mean for G
        self.K_G = None   # Posterior Column Covariance for G
        self.Psi_G = None # Posterior Scale Matrix for Noise
        self.nu_G = None  # Posterior Degrees of Freedom for Noise

    def _initialize_hyperparameters(self, state_dim, control_dim):
        """Initializes the prior hyperparameters."""
        aug_dim = state_dim * (1 + control_dim)
        self.M_G = np.zeros((state_dim, aug_dim))
        self.K_G = np.identity(aug_dim) * 1e6
        self.Psi_G = np.identity(state_dim) * 1e-3
        self.nu_G = float(state_dim)

    def _update_hyperparameters(self, xi, x_prime):
        """Performs a rank-1 update of the posterior."""
        K_inv = np.linalg.inv(self.K_G)
        K_inv = self.lam * K_inv + np.outer(xi, xi)
        self.K_G = np.linalg.inv(K_inv)
        
        prediction_error = x_prime - self.M_G @ xi
        gain = self.K_G @ xi
        self.M_G = self.M_G + np.outer(prediction_error, gain)
        
        self.nu_G = self.lam * self.nu_G + 1
        self.Psi_G = self.lam * self.Psi_G + np.outer(prediction_error, prediction_error)

    def fit(self):
        """Overrides the parent fit method for Bayesian incremental learning."""
        state_dim, num_snapshots = self.X1.shape
        control_dim = self.U.shape[0]

        self._initialize_hyperparameters(state_dim, control_dim)
        self.history = {'M_G': [], 'eigenvalues': []}

        print(f"Starting Bayesian Incremental Fit. Total Snapshots: {num_snapshots}")
        for k in range(num_snapshots):
            x_k = self.X1[:, k]
            u_k = self.U[:, k]
            x_prime_k = self.X2[:, k]
            xi_k = np.concatenate([x_k, np.kron(u_k, x_k)])
            self._update_hyperparameters(xi_k, x_prime_k)

        self.G = self.M_G
        self.A = self.G[:, :state_dim]
        self.B = self.G[:, state_dim:]
        
        U_r, _, _ = np.linalg.svd(self.X1, full_matrices=False)
        U_r = U_r[:, :self.rank]
        A_reduced = U_r.T @ self.A @ U_r
        self.eigs, _ = np.linalg.eig(A_reduced)
        
        print("Bayesian fit complete.")
        return self
    
    def reconstruction_error(self):
        """
        Calculates the Frobenius norm of the reconstruction error.
        This method is added directly to ensure it exists.
        """
        if self.A is None or self.B is None:
            raise ValueError("The model must be fitted before calculating the reconstruction error.")
        
        # Reconstruct the X2 matrix using the posterior mean operators
        X_prime_reconstructed = self.A.dot(self.X1) + self.B.dot(self.U * self.X1)
        
        # Calculate and return the Frobenius norm of the difference
        return np.linalg.norm(self.X2 - X_prime_reconstructed, 'fro')

    def sample_posterior_G(self, n_samples=100):
        """Draws samples of the G matrix from the learned posterior."""
        from scipy.stats import invwishart
        noise_cov_samples = invwishart.rvs(df=self.nu_G, scale=self.Psi_G, size=n_samples)
        if n_samples == 1:
            noise_cov_samples = [noise_cov_samples]

        G_samples = []
        for Sigma_sample in noise_cov_samples:
            mean_vec = self.M_G.flatten()
            Sigma_sample_2d = np.atleast_2d(Sigma_sample)
            cov_mat = np.kron(self.K_G, Sigma_sample_2d)
            
            g_vec_sample = np.random.multivariate_normal(mean_vec, cov_mat)
            G_sample = g_vec_sample.reshape(self.M_G.shape)
            G_samples.append(G_sample)
            
        return G_samples