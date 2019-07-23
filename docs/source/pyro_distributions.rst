Pyro-Compatible Distributions
-----------------------------
This interface provides a number of PyTorch-style distributions that use
funsors internally to perform inference. These high-level objects are based on
two wrapping classes: :class:`~funsor.pyro_distributions.FunsorDistribution`
which wraps a funsor in a PyTorch-distributions-compatible interface, and
:class:`~funsor.pyro_distributions.DistributionFunsor` which wraps a
PyTorch-style distribution in a funsor-compatible interface.

:class:`~funsor.pyro_distributions.FunsorDistribution` objects can be used
directly in Pyro models (using the standard Pyro backend).

.. automodule:: funsor.pyro_distributions
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
