Pyro-Compatible Distributions
-----------------------------
This interface provides a number of PyTorch-style distributions that use
funsors internally to perform inference. These high-level objects are based on
a wrapping class: :class:`~funsor.pyro_distributions.FunsorDistribution` which
wraps a funsor in a PyTorch-distributions-compatible interface.
:class:`~funsor.pyro.distribution.FunsorDistribution` objects can be used
directly in Pyro models (using the standard Pyro backend).

FunsorDistribution Base Class
=============================
.. automodule:: funsor.pyro.distribution
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Hidden Markov Models
====================
.. automodule:: funsor.pyro.hmm
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Conversion Utilities
====================
.. automodule:: funsor.pyro.convert
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
