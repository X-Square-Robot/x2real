"""PolicyImplementation package — one ManaPolicy component (required).

Each protocol family (wallx / openpi / gr00t / replay / ...) provides
a concrete :class:`~manaenv.policy.policy_implementation.base.PolicyImplementationBase`
subclass discovered by the global lazy registry. The translator selects
one from the policy yaml section and instantiates it via the standard
``cfg.class_type()`` + ``.load(cfg)`` pattern.

Use explicit submodule imports (no aggregated re-exports here per
policy plugin import contract).
"""
