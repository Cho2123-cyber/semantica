"""
Pipeline Composition Module

This module composes existing pipelines into larger ones, turning any built
:class:`~semantica.pipeline.pipeline_builder.Pipeline` into a reusable
sub-pipeline. Composition copies the source pipelines, namespaces their step
names to avoid collisions, rewires dependencies across the seam, and validates
the result before returning it.

Key Features:
    - Sequential composition (``chain``): run pipelines one after another
    - Parallel composition (``merge``): run pipelines side by side, with an
      optional join step that waits for all of them
    - Nested composition (``nest``): splice a pipeline in before, after, or in
      place of a single step of another pipeline
    - Automatic step-name namespacing with explicit collision detection
    - Source pipelines are never mutated: every composition returns new objects
    - Composed pipelines record their lineage in ``metadata["composition"]``
      and stay serializable through :class:`PipelineSerializer`

Main Classes:
    - PipelineComposer: Pipeline composition engine

Example Usage:
    >>> from semantica.pipeline import PipelineComposer
    >>> composer = PipelineComposer()
    >>> full = composer.chain(ingest_pipeline, kg_pipeline, name="end_to_end")
    >>> [step.name for step in full.steps]
    ['ingest.load', 'ingest.parse', 'kg.extract', 'kg.store']

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional, Sequence, Union

from ..utils.exceptions import ValidationError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker
from .pipeline_builder import Pipeline, PipelineStep, StepStatus

#: Separator inserted between a namespace and the original step name.
DEFAULT_SEPARATOR = "."

#: Nesting modes accepted by :meth:`PipelineComposer.nest`.
NEST_MODES = ("after", "before", "replace")


def copy_step(
    step: PipelineStep,
    name: Optional[str] = None,
    dependencies: Optional[List[str]] = None,
) -> PipelineStep:
    """
    Copy a pipeline step for use in a composed pipeline.

    The copy is structural, not deep: ``config`` is copied one level (so
    rewiring a composed pipeline never mutates the source step's config dict)
    while the values inside it — handlers, clients, stores — are shared with
    the original. Runtime state (``status``, ``result``, ``error``) is reset,
    because a composed pipeline is a fresh definition rather than a resumed run.

    Args:
        step: Step to copy
        name: Replacement step name (defaults to the original name)
        dependencies: Replacement dependency list (defaults to a copy of the
            original dependencies)

    Returns:
        New PipelineStep instance
    """
    new_dependencies = (
        list(step.dependencies) if dependencies is None else list(dependencies)
    )
    new_config = dict(step.config)

    # PipelineBuilder.add_step() leaves a "dependencies" key inside step.config,
    # and ExecutionEngine forwards config to the handler as kwargs. Keep the two
    # in sync so a rewired step never hands its handler stale dependency names.
    if "dependencies" in new_config:
        new_config["dependencies"] = list(new_dependencies)

    return PipelineStep(
        name=step.name if name is None else name,
        step_type=step.step_type,
        config=new_config,
        dependencies=new_dependencies,
        handler=step.handler,
        status=StepStatus.PENDING,
        result=None,
        error=None,
        delta_mode=getattr(step, "delta_mode", False),
        base_version_id=getattr(step, "base_version_id", None),
        target_version_id=getattr(step, "target_version_id", None),
    )


def namespace_steps(
    steps: Sequence[PipelineStep],
    namespace: Optional[str] = None,
    separator: str = DEFAULT_SEPARATOR,
) -> List[PipelineStep]:
    """
    Copy steps under a namespace prefix, rewriting internal dependencies.

    Only dependencies that resolve to a step inside ``steps`` are rewritten. A
    dependency that points outside the group is left untouched so the caller
    (or the validator) can see it is dangling.

    Args:
        steps: Steps to namespace
        namespace: Prefix to apply; ``None`` or ``""`` copies without renaming
        separator: Separator between the namespace and the step name

    Returns:
        New list of copied steps
    """
    known = {step.name for step in steps}

    def rename(step_name: str) -> str:
        if not namespace:
            return step_name
        return f"{namespace}{separator}{step_name}"

    return [
        copy_step(
            step,
            name=rename(step.name),
            dependencies=[
                rename(dep) if dep in known else dep for dep in step.dependencies
            ],
        )
        for step in steps
    ]


def entry_steps(steps: Sequence[PipelineStep]) -> List[PipelineStep]:
    """
    Return the steps that start a group: those with no dependency inside it.

    Args:
        steps: Steps to inspect

    Returns:
        Entry steps, in their original order
    """
    known = {step.name for step in steps}
    return [
        step for step in steps if not any(dep in known for dep in step.dependencies)
    ]


def terminal_steps(steps: Sequence[PipelineStep]) -> List[PipelineStep]:
    """
    Return the steps that end a group: those nothing inside it depends on.

    Args:
        steps: Steps to inspect

    Returns:
        Terminal steps, in their original order
    """
    depended_on = {dep for step in steps for dep in step.dependencies}
    return [step for step in steps if step.name not in depended_on]


def _extend_dependencies(step: PipelineStep, names: Sequence[str]) -> None:
    """Append dependency names to a step, in order, without duplicating them."""
    for name in names:
        if name not in step.dependencies:
            step.dependencies.append(name)
    if "dependencies" in step.config:
        step.config["dependencies"] = list(step.dependencies)


def _replace_dependency(
    step: PipelineStep, old: str, replacements: Sequence[str]
) -> None:
    """Swap one dependency name for a set of names, preserving order."""
    if old not in step.dependencies:
        return

    rewired: List[str] = []
    for dep in step.dependencies:
        candidates = replacements if dep == old else [dep]
        for candidate in candidates:
            if candidate not in rewired:
                rewired.append(candidate)

    step.dependencies = rewired
    if "dependencies" in step.config:
        step.config["dependencies"] = list(rewired)


class PipelineComposer:
    """
    Pipeline composition engine.

    • Chains pipelines so one runs after another
    • Merges pipelines so they run side by side, optionally joining at the end
    • Nests a pipeline before, after, or in place of a single step
    • Namespaces step names and rewires dependencies across the seam
    • Leaves source pipelines untouched and records composition lineage
    • Validates every composed pipeline before returning it
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize pipeline composer.

        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options, forwarded to the
                internal PipelineValidator
        """
        self.logger = get_logger("pipeline_composer")
        self.config = config or {}
        self.config.update(kwargs)

        self.progress_tracker = get_progress_tracker()
        if not self.progress_tracker.enabled:
            self.progress_tracker.enabled = True

        from .pipeline_validator import PipelineValidator

        self.validator = PipelineValidator(**self.config)

    def chain(
        self,
        *pipelines: Pipeline,
        name: Optional[str] = None,
        namespace: Union[bool, Sequence[Optional[str]]] = True,
        separator: str = DEFAULT_SEPARATOR,
        config: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> Pipeline:
        """
        Compose pipelines sequentially.

        Every entry step of each pipeline is made to depend on every terminal
        step of the pipeline before it, so a pipeline only starts once the
        previous one has fully finished.

        ``ExecutionEngine`` threads a single value through the topological
        order, so when a pipeline has several terminal steps the next pipeline
        receives the output of whichever terminal ran last. Give a pipeline a
        single terminal step when the hand-off value matters.

        Args:
            *pipelines: Pipelines to chain, in execution order (at least one)
            name: Name of the composed pipeline; defaults to the source names
                joined with "+"
            namespace: ``True`` to prefix each pipeline's steps with its own
                name, ``False`` to keep step names as they are, or a sequence
                of explicit prefixes (one per pipeline, ``None`` for no prefix)
            separator: Separator between the namespace and the step name
            config: Configuration overrides applied on top of the merged
                source configurations
            validate: Validate the composed pipeline before returning it

        Returns:
            Composed pipeline

        Raises:
            ValidationError: No pipelines given, a source pipeline is empty,
                step names collide, or the composed pipeline fails validation
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="pipeline",
            submodule="PipelineComposer",
            message=f"Chaining {len(pipelines)} pipelines",
        )

        try:
            self._require_pipelines(pipelines, "chain")
            namespaces = self._resolve_namespaces(pipelines, namespace)

            self.progress_tracker.update_tracking(
                tracking_id, message="Namespacing and rewiring steps..."
            )

            steps: List[PipelineStep] = []
            previous_terminals: List[str] = []
            for pipeline, prefix in zip(pipelines, namespaces):
                group = namespace_steps(pipeline.steps, prefix, separator)
                for step in entry_steps(group):
                    _extend_dependencies(step, previous_terminals)
                previous_terminals = [step.name for step in terminal_steps(group)]
                steps.extend(group)

            composed = self._assemble(
                operation="chain",
                name=name or self._default_name(pipelines),
                steps=steps,
                pipelines=pipelines,
                namespaces=namespaces,
                config=config,
                validate=validate,
            )

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=(
                    f"Chained {len(pipelines)} pipelines into "
                    f"'{composed.name}' ({len(composed.steps)} steps)"
                ),
            )
            return composed

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def merge(
        self,
        *pipelines: Pipeline,
        name: Optional[str] = None,
        namespace: Union[bool, Sequence[Optional[str]]] = True,
        separator: str = DEFAULT_SEPARATOR,
        join: Optional[PipelineStep] = None,
        config: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> Pipeline:
        """
        Compose pipelines side by side.

        No dependencies are added between the sources, so they stay independent
        branches of one graph and ``ParallelismManager`` is free to run them
        concurrently. Pass ``join`` to add a step that waits for every branch.

        Args:
            *pipelines: Pipelines to merge (at least one)
            name: Name of the composed pipeline; defaults to the source names
                joined with "+"
            namespace: ``True`` to prefix each pipeline's steps with its own
                name, ``False`` to keep step names as they are, or a sequence
                of explicit prefixes (one per pipeline, ``None`` for no prefix)
            separator: Separator between the namespace and the step name
            join: Optional step appended after all branches; it is copied, and
                its dependencies are extended with every branch's terminal
                steps. It is never namespaced — it belongs to the composition,
                not to any single source
            config: Configuration overrides applied on top of the merged
                source configurations
            validate: Validate the composed pipeline before returning it

        Returns:
            Composed pipeline

        Raises:
            ValidationError: No pipelines given, a source pipeline is empty,
                step names collide, or the composed pipeline fails validation
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="pipeline",
            submodule="PipelineComposer",
            message=f"Merging {len(pipelines)} pipelines",
        )

        try:
            self._require_pipelines(pipelines, "merge")
            namespaces = self._resolve_namespaces(pipelines, namespace)

            self.progress_tracker.update_tracking(
                tracking_id, message="Namespacing branches..."
            )

            steps: List[PipelineStep] = []
            branch_terminals: List[str] = []
            for pipeline, prefix in zip(pipelines, namespaces):
                group = namespace_steps(pipeline.steps, prefix, separator)
                branch_terminals.extend(step.name for step in terminal_steps(group))
                steps.extend(group)

            if join is not None:
                join_step = copy_step(join)
                _extend_dependencies(join_step, branch_terminals)
                steps.append(join_step)

            composed = self._assemble(
                operation="merge",
                name=name or self._default_name(pipelines),
                steps=steps,
                pipelines=pipelines,
                namespaces=namespaces,
                config=config,
                validate=validate,
                extra_metadata={"join": join.name if join is not None else None},
            )

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=(
                    f"Merged {len(pipelines)} pipelines into "
                    f"'{composed.name}' ({len(composed.steps)} steps)"
                ),
            )
            return composed

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    def nest(
        self,
        parent: Pipeline,
        child: Pipeline,
        at: str,
        mode: str = "after",
        name: Optional[str] = None,
        namespace: Union[bool, str, None] = True,
        separator: str = DEFAULT_SEPARATOR,
        config: Optional[Dict[str, Any]] = None,
        validate: bool = True,
    ) -> Pipeline:
        """
        Splice a pipeline into a single position of another pipeline.

        The parent keeps its own step names; only the child is namespaced, so
        the same sub-pipeline can be nested at several positions under
        different prefixes.

        Modes:
            ``"after"``: the child runs after ``at``. Steps that depended on
                ``at`` now depend on the child's terminal steps instead.
            ``"before"``: the child runs before ``at``. The child inherits
                ``at``'s dependencies and ``at`` waits for the child.
            ``"replace"``: the child takes the place of ``at``, which is
                removed. The child inherits ``at``'s dependencies and ``at``'s
                dependents now wait for the child.

        Args:
            parent: Pipeline to splice into
            child: Pipeline to insert
            at: Name of the parent step to insert at
            mode: One of ``"after"``, ``"before"``, ``"replace"``
            name: Name of the composed pipeline; defaults to the parent's name
            namespace: ``True`` to prefix the child's steps with the child
                pipeline's name, a string for an explicit prefix, or ``False``
                / ``None`` to keep the child's step names as they are
            separator: Separator between the namespace and the step name
            config: Configuration overrides applied on top of the merged
                source configurations
            validate: Validate the composed pipeline before returning it

        Returns:
            Composed pipeline

        Raises:
            ValidationError: Unknown mode, empty pipeline, ``at`` not found in
                the parent, step names collide, or the composed pipeline fails
                validation
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="pipeline",
            submodule="PipelineComposer",
            message=f"Nesting '{child.name}' {mode} '{at}' in '{parent.name}'",
        )

        try:
            if mode not in NEST_MODES:
                raise ValidationError(
                    f"Unknown nest mode {mode!r}; expected one of "
                    f"{', '.join(repr(m) for m in NEST_MODES)}"
                )
            self._require_pipelines((parent, child), "nest")

            anchor_index = next(
                (i for i, s in enumerate(parent.steps) if s.name == at), None
            )
            if anchor_index is None:
                raise ValidationError(
                    f"Step '{at}' not found in pipeline '{parent.name}'"
                )

            self.progress_tracker.update_tracking(
                tracking_id, message="Rewiring parent around the nested pipeline..."
            )

            prefix = child.name if namespace is True else (namespace or None)
            nested = namespace_steps(child.steps, prefix, separator)
            nested_entries = entry_steps(nested)
            nested_terminals = [step.name for step in terminal_steps(nested)]

            parent_steps = namespace_steps(parent.steps, None, separator)
            anchor = parent_steps[anchor_index]

            if mode == "after":
                for step in nested_entries:
                    _extend_dependencies(step, [anchor.name])
                for step in parent_steps:
                    if step is not anchor:
                        _replace_dependency(step, anchor.name, nested_terminals)
                parent_steps[anchor_index + 1 : anchor_index + 1] = nested
            elif mode == "before":
                for step in nested_entries:
                    _extend_dependencies(step, anchor.dependencies)
                anchor.dependencies = []
                if "dependencies" in anchor.config:
                    anchor.config["dependencies"] = []
                _extend_dependencies(anchor, nested_terminals)
                parent_steps[anchor_index:anchor_index] = nested
            else:  # "replace"
                for step in nested_entries:
                    _extend_dependencies(step, anchor.dependencies)
                for step in parent_steps:
                    if step is not anchor:
                        _replace_dependency(step, anchor.name, nested_terminals)
                parent_steps[anchor_index : anchor_index + 1] = nested

            composed = self._assemble(
                operation="nest",
                name=name or parent.name,
                steps=parent_steps,
                pipelines=(parent, child),
                namespaces=[None, prefix],
                config=config,
                validate=validate,
                extra_metadata={"at": at, "mode": mode},
            )

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=(
                    f"Nested '{child.name}' {mode} '{at}' in "
                    f"'{composed.name}' ({len(composed.steps)} steps)"
                ),
            )
            return composed

        except Exception as e:
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_pipelines(pipelines: Sequence[Pipeline], operation: str) -> None:
        """Reject compositions that cannot produce a well-formed pipeline."""
        if not pipelines:
            raise ValidationError(f"{operation}() requires at least one pipeline")

        for pipeline in pipelines:
            if not getattr(pipeline, "steps", None):
                raise ValidationError(
                    f"Cannot {operation} pipeline '{getattr(pipeline, 'name', '?')}': "
                    f"it has no steps"
                )
            # A pipeline whose steps form a cycle has no entry and no terminal
            # step, which would silently drop the dependency link at the seam.
            if not entry_steps(pipeline.steps) or not terminal_steps(pipeline.steps):
                raise ValidationError(
                    f"Cannot {operation} pipeline '{pipeline.name}': its steps "
                    f"form a cycle, so it has no entry or terminal step"
                )

    @staticmethod
    def _default_name(pipelines: Sequence[Pipeline]) -> str:
        """Derive a composed pipeline name from the source names."""
        return "+".join(pipeline.name for pipeline in pipelines)

    @staticmethod
    def _resolve_namespaces(
        pipelines: Sequence[Pipeline],
        namespace: Union[bool, Sequence[Optional[str]]],
    ) -> List[Optional[str]]:
        """
        Turn the ``namespace`` argument into one prefix per pipeline.

        ``True`` derives prefixes from the pipeline names, disambiguating
        repeats with a numeric suffix so a pipeline can be composed with
        itself. ``False`` disables prefixing. A sequence is used verbatim and
        must have one entry per pipeline.
        """
        if namespace is True:
            resolved: List[Optional[str]] = []
            used: Dict[str, int] = {}
            for pipeline in pipelines:
                base = pipeline.name
                used[base] = used.get(base, 0) + 1
                resolved.append(base if used[base] == 1 else f"{base}_{used[base]}")
            return resolved

        if namespace is False or namespace is None:
            return [None] * len(pipelines)

        if isinstance(namespace, str):
            raise ValidationError(
                "namespace must be True, False, or one prefix per pipeline; "
                "a single string would give every pipeline the same prefix"
            )

        prefixes = list(namespace)
        if len(prefixes) != len(pipelines):
            raise ValidationError(
                f"namespace has {len(prefixes)} prefixes but "
                f"{len(pipelines)} pipelines were given"
            )
        return [prefix or None for prefix in prefixes]

    def _assemble(
        self,
        operation: str,
        name: str,
        steps: List[PipelineStep],
        pipelines: Sequence[Pipeline],
        namespaces: Sequence[Optional[str]],
        config: Optional[Dict[str, Any]],
        validate: bool,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Pipeline:
        """Check for collisions, build the Pipeline, and validate it."""
        self._reject_duplicate_names(steps, operation)

        merged_config: Dict[str, Any] = {}
        for pipeline in pipelines:
            merged_config.update(pipeline.config or {})
        if config:
            merged_config.update(config)

        composition: Dict[str, Any] = {
            "operation": operation,
            "sources": [
                {
                    "name": pipeline.name,
                    "namespace": prefix,
                    "step_count": len(pipeline.steps),
                }
                for pipeline, prefix in zip(pipelines, namespaces)
            ],
        }
        if extra_metadata:
            composition.update(extra_metadata)

        composed = Pipeline(
            name=name,
            steps=steps,
            config=merged_config,
            metadata={
                "step_count": len(steps),
                "parallelism": merged_config.get("parallelism", 1),
                "composition": composition,
            },
        )

        if validate:
            result = self.validator.validate_pipeline(composed)
            for warning in result.warnings:
                self.logger.debug(f"Composition warning ({name}): {warning}")
            if not result.valid:
                raise ValidationError(
                    f"Composed pipeline '{name}' failed validation: {result.errors}"
                )

        self.logger.info(
            f"Composed pipeline '{name}' via {operation} "
            f"from {len(pipelines)} source(s) with {len(steps)} steps"
        )
        return composed

    @staticmethod
    def _reject_duplicate_names(steps: Sequence[PipelineStep], operation: str) -> None:
        """Raise if composition produced two steps with the same name."""
        seen = set()
        duplicates = []
        for step in steps:
            if step.name in seen and step.name not in duplicates:
                duplicates.append(step.name)
            seen.add(step.name)

        if duplicates:
            raise ValidationError(
                f"Cannot {operation} pipelines: duplicate step names {duplicates}. "
                f"Use namespace=True, or pass explicit prefixes, to keep them apart."
            )
