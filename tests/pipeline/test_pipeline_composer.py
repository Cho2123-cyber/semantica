"""Tests for pipeline composition (chain / merge / nest and builder.include)."""

import unittest
from unittest.mock import MagicMock, patch

from semantica.pipeline.execution_engine import ExecutionEngine
from semantica.pipeline.pipeline_builder import (
    Pipeline,
    PipelineBuilder,
    PipelineSerializer,
    PipelineStep,
    StepStatus,
)
from semantica.pipeline.pipeline_composer import (
    PipelineComposer,
    entry_steps,
    terminal_steps,
)
from semantica.utils.exceptions import ValidationError


def linear_pipeline(name, steps, **config):
    """Build a pipeline whose steps run one after another."""
    builder = PipelineBuilder()
    previous = None
    for step_name, handler in steps:
        builder.add_step(step_name, "math", handler=handler)
        if previous is not None:
            builder.connect_steps(previous, step_name)
        previous = step_name
    for key, value in config.items():
        builder.pipeline_config[key] = value
    return builder.build(name)


def deps_by_name(pipeline):
    """Map step name -> dependency list, for concise assertions."""
    return {step.name: list(step.dependencies) for step in pipeline.steps}


class PipelineComposerTestCase(unittest.TestCase):
    """Shared fixtures for composition tests."""

    def setUp(self):
        self.mock_tracker_patcher = patch(
            "semantica.utils.progress_tracker.get_progress_tracker"
        )
        self.mock_get_tracker = self.mock_tracker_patcher.start()
        self.mock_get_tracker.return_value = MagicMock()

        self.composer = PipelineComposer()
        self.alpha = linear_pipeline(
            "alpha",
            [
                ("load", lambda data, **kwargs: data + 1),
                ("parse", lambda data, **kwargs: data * 2),
            ],
        )
        self.beta = linear_pipeline(
            "beta", [("load", lambda data, **kwargs: data + 10)]
        )

    def tearDown(self):
        self.mock_tracker_patcher.stop()


class TestGroupHelpers(PipelineComposerTestCase):
    """entry_steps / terminal_steps identify the seams of a step group."""

    def test_entry_and_terminal_steps_of_a_linear_pipeline(self):
        self.assertEqual([s.name for s in entry_steps(self.alpha.steps)], ["load"])
        self.assertEqual([s.name for s in terminal_steps(self.alpha.steps)], ["parse"])

    def test_dependency_outside_the_group_still_counts_as_an_entry(self):
        steps = [PipelineStep(name="a", step_type="t", dependencies=["elsewhere"])]
        self.assertEqual([s.name for s in entry_steps(steps)], ["a"])


class TestChain(PipelineComposerTestCase):
    """chain() runs pipelines one after another."""

    def test_namespaces_steps_and_links_the_seam(self):
        chained = self.composer.chain(self.alpha, self.beta, name="chained")

        self.assertEqual(chained.name, "chained")
        self.assertEqual(
            [s.name for s in chained.steps],
            ["alpha.load", "alpha.parse", "beta.load"],
        )
        self.assertEqual(
            deps_by_name(chained),
            {
                "alpha.load": [],
                "alpha.parse": ["alpha.load"],
                "beta.load": ["alpha.parse"],
            },
        )

    def test_default_name_joins_source_names(self):
        self.assertEqual(self.composer.chain(self.alpha, self.beta).name, "alpha+beta")

    def test_executes_end_to_end_in_composed_order(self):
        chained = self.composer.chain(self.alpha, self.beta)

        result = ExecutionEngine().execute_pipeline(chained, data=1)

        self.assertTrue(result.success)
        self.assertEqual(result.output, 14)  # ((1 + 1) * 2) + 10
        self.assertTrue(
            all(step.status == StepStatus.COMPLETED for step in chained.steps)
        )

    def test_source_pipelines_are_not_mutated(self):
        chained = self.composer.chain(self.alpha, self.beta)
        ExecutionEngine().execute_pipeline(chained, data=1)

        self.assertEqual([s.name for s in self.alpha.steps], ["load", "parse"])
        self.assertEqual(deps_by_name(self.alpha), {"load": [], "parse": ["load"]})
        self.assertTrue(
            all(step.status == StepStatus.PENDING for step in self.alpha.steps)
        )

    def test_handlers_are_carried_over(self):
        chained = self.composer.chain(self.alpha)
        for step in chained.steps:
            self.assertIsNotNone(step.handler)

    def test_single_pipeline_returns_a_namespaced_copy(self):
        copied = self.composer.chain(self.alpha, name="only")

        self.assertEqual([s.name for s in copied.steps], ["alpha.load", "alpha.parse"])
        self.assertIsNot(copied.steps[0], self.alpha.steps[0])

    def test_explicit_namespaces_are_used_verbatim(self):
        chained = self.composer.chain(self.alpha, self.beta, namespace=["first", None])

        self.assertEqual(
            [s.name for s in chained.steps],
            ["first.load", "first.parse", "load"],
        )

    def test_custom_separator(self):
        chained = self.composer.chain(self.alpha, separator="__")
        self.assertEqual(chained.steps[0].name, "alpha__load")

    def test_same_pipeline_twice_gets_distinct_namespaces(self):
        chained = self.composer.chain(self.alpha, self.alpha)

        self.assertEqual(
            [s.name for s in chained.steps],
            ["alpha.load", "alpha.parse", "alpha_2.load", "alpha_2.parse"],
        )
        self.assertEqual(chained.steps[2].dependencies, ["alpha.parse"])

    def test_collision_without_namespacing_is_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            self.composer.chain(self.alpha, self.beta, namespace=False)

        self.assertIn("duplicate step names", str(ctx.exception))
        self.assertIn("load", str(ctx.exception))

    def test_disjoint_step_names_may_skip_namespacing(self):
        gamma = linear_pipeline("gamma", [("store", lambda data, **kw: data)])

        chained = self.composer.chain(self.alpha, gamma, namespace=False)

        self.assertEqual([s.name for s in chained.steps], ["load", "parse", "store"])
        self.assertEqual(chained.steps[2].dependencies, ["parse"])

    def test_requires_at_least_one_pipeline(self):
        with self.assertRaises(ValidationError):
            self.composer.chain()

    def test_rejects_empty_pipeline(self):
        with self.assertRaises(ValidationError) as ctx:
            self.composer.chain(self.alpha, Pipeline(name="empty"))

        self.assertIn("no steps", str(ctx.exception))

    def test_rejects_cyclic_pipeline(self):
        cyclic = Pipeline(
            name="cyclic",
            steps=[
                PipelineStep(name="a", step_type="t", dependencies=["b"]),
                PipelineStep(name="b", step_type="t", dependencies=["a"]),
            ],
        )

        with self.assertRaises(ValidationError) as ctx:
            self.composer.chain(cyclic, self.alpha)

        self.assertIn("cycle", str(ctx.exception))

    def test_rejects_a_bare_string_namespace(self):
        with self.assertRaises(ValidationError) as ctx:
            self.composer.chain(self.alpha, self.beta, namespace="shared")

        self.assertIn("one prefix per pipeline", str(ctx.exception))

    def test_rejects_namespace_length_mismatch(self):
        with self.assertRaises(ValidationError) as ctx:
            self.composer.chain(self.alpha, self.beta, namespace=["only_one"])

        self.assertIn("1 prefixes", str(ctx.exception))

    def test_invalid_composition_is_rejected_by_validation(self):
        dangling = Pipeline(
            name="dangling",
            steps=[
                PipelineStep(name="a", step_type="t", dependencies=["ghost"]),
            ],
        )

        with self.assertRaises(ValidationError) as ctx:
            self.composer.chain(dangling, self.alpha)

        self.assertIn("failed validation", str(ctx.exception))

    def test_validate_false_skips_validation(self):
        dangling = Pipeline(
            name="dangling",
            steps=[
                PipelineStep(name="a", step_type="t", dependencies=["ghost"]),
            ],
        )

        composed = self.composer.chain(dangling, self.alpha, validate=False)

        self.assertEqual(composed.steps[0].dependencies, ["ghost"])


class TestMerge(PipelineComposerTestCase):
    """merge() runs pipelines side by side."""

    def test_branches_stay_independent(self):
        merged = self.composer.merge(self.alpha, self.beta, name="merged")

        self.assertEqual(
            deps_by_name(merged),
            {
                "alpha.load": [],
                "alpha.parse": ["alpha.load"],
                "beta.load": [],
            },
        )

    def test_join_step_waits_for_every_branch_terminal(self):
        join = PipelineStep(
            name="join", step_type="join", handler=lambda data, **kwargs: data
        )

        merged = self.composer.merge(self.alpha, self.beta, join=join)

        self.assertEqual(merged.steps[-1].name, "join")
        self.assertEqual(merged.steps[-1].dependencies, ["alpha.parse", "beta.load"])
        self.assertEqual(merged.metadata["composition"]["join"], "join")

    def test_join_step_is_copied_not_mutated(self):
        join = PipelineStep(name="join", step_type="join")

        self.composer.merge(self.alpha, self.beta, join=join)

        self.assertEqual(join.dependencies, [])

    def test_join_is_not_namespaced(self):
        join = PipelineStep(name="join", step_type="join")

        merged = self.composer.merge(self.alpha, join=join)

        self.assertEqual([s.name for s in merged.steps][-1], "join")

    def test_join_name_collision_is_rejected(self):
        join = PipelineStep(name="alpha.load", step_type="join")

        with self.assertRaises(ValidationError) as ctx:
            self.composer.merge(self.alpha, join=join)

        self.assertIn("duplicate step names", str(ctx.exception))

    def test_merged_pipeline_executes(self):
        join = PipelineStep(
            name="join", step_type="join", handler=lambda data, **kwargs: data * 100
        )

        merged = self.composer.merge(self.alpha, self.beta, join=join)
        result = ExecutionEngine().execute_pipeline(merged, data=1)

        self.assertTrue(result.success)
        self.assertTrue(
            all(step.status == StepStatus.COMPLETED for step in merged.steps)
        )


class TestNest(PipelineComposerTestCase):
    """nest() splices a pipeline into one position of another."""

    def setUp(self):
        super().setUp()
        # load fans out to two independent branches that both converge on store.
        builder = PipelineBuilder()
        for name in ("load", "left", "right", "store"):
            builder.add_step(name, "math", handler=lambda data, **kwargs: data)
        builder.connect_steps("load", "left")
        builder.connect_steps("load", "right")
        builder.connect_steps("left", "store")
        builder.connect_steps("right", "store")
        self.fanout = builder.build("fanout")

    def test_after_rewires_every_dependent_of_the_anchor(self):
        nested = self.composer.nest(self.fanout, self.beta, at="load")

        self.assertEqual(
            [s.name for s in nested.steps],
            ["load", "beta.load", "left", "right", "store"],
        )
        self.assertEqual(
            deps_by_name(nested),
            {
                "load": [],
                "beta.load": ["load"],
                "left": ["beta.load"],
                "right": ["beta.load"],
                "store": ["left", "right"],
            },
        )

    def test_before_inherits_the_anchor_dependencies(self):
        nested = self.composer.nest(self.fanout, self.beta, at="store", mode="before")

        self.assertEqual(nested.steps[-2].name, "beta.load")
        self.assertEqual(nested.steps[-2].dependencies, ["left", "right"])
        self.assertEqual(deps_by_name(nested)["store"], ["beta.load"])

    def test_replace_drops_the_anchor_and_rewires_both_sides(self):
        nested = self.composer.nest(self.fanout, self.beta, at="left", mode="replace")

        self.assertNotIn("left", [s.name for s in nested.steps])
        self.assertEqual(
            deps_by_name(nested),
            {
                "load": [],
                "beta.load": ["load"],
                "right": ["load"],
                "store": ["beta.load", "right"],
            },
        )

    def test_parent_step_names_are_preserved(self):
        nested = self.composer.nest(self.fanout, self.beta, at="load")

        for name in ("load", "left", "right", "store"):
            self.assertIn(name, [s.name for s in nested.steps])

    def test_explicit_child_namespace(self):
        nested = self.composer.nest(self.fanout, self.beta, at="load", namespace="sub")

        self.assertIn("sub.load", [s.name for s in nested.steps])

    def test_child_namespace_can_be_disabled_when_names_are_disjoint(self):
        gamma = linear_pipeline("gamma", [("enrich", lambda data, **kw: data)])

        nested = self.composer.nest(self.fanout, gamma, at="load", namespace=False)

        self.assertIn("enrich", [s.name for s in nested.steps])

    def test_child_name_collision_is_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            self.composer.nest(self.fanout, self.beta, at="load", namespace=False)

        self.assertIn("duplicate step names", str(ctx.exception))

    def test_same_child_nested_twice_under_different_prefixes(self):
        once = self.composer.nest(self.fanout, self.beta, at="left", namespace="first")
        twice = self.composer.nest(once, self.beta, at="right", namespace="second")

        self.assertIn("first.load", [s.name for s in twice.steps])
        self.assertIn("second.load", [s.name for s in twice.steps])

    def test_unknown_anchor_is_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            self.composer.nest(self.fanout, self.beta, at="nowhere")

        self.assertIn("not found", str(ctx.exception))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            self.composer.nest(self.fanout, self.beta, at="load", mode="sideways")

        self.assertIn("Unknown nest mode", str(ctx.exception))

    def test_nested_pipeline_executes_in_order(self):
        seen = []

        def record(label):
            def handler(data, **kwargs):
                seen.append(label)
                return data

            return handler

        parent = linear_pipeline(
            "parent", [("first", record("first")), ("last", record("last"))]
        )
        child = linear_pipeline("child", [("middle", record("middle"))])

        nested = self.composer.nest(parent, child, at="first")
        result = ExecutionEngine().execute_pipeline(nested, data=0)

        self.assertTrue(result.success)
        self.assertEqual(seen, ["first", "middle", "last"])

    def test_source_pipelines_are_not_mutated(self):
        self.composer.nest(self.fanout, self.beta, at="load", mode="replace")

        self.assertEqual(
            [s.name for s in self.fanout.steps],
            ["load", "left", "right", "store"],
        )
        self.assertEqual(deps_by_name(self.fanout)["left"], ["load"])
        self.assertEqual([s.name for s in self.beta.steps], ["load"])


class TestCompositionMetadataAndConfig(PipelineComposerTestCase):
    """Composed pipelines record lineage and stay serializable."""

    def test_metadata_records_the_composition_lineage(self):
        chained = self.composer.chain(self.alpha, self.beta, name="chained")

        composition = chained.metadata["composition"]
        self.assertEqual(composition["operation"], "chain")
        self.assertEqual(
            composition["sources"],
            [
                {"name": "alpha", "namespace": "alpha", "step_count": 2},
                {"name": "beta", "namespace": "beta", "step_count": 1},
            ],
        )
        self.assertEqual(chained.metadata["step_count"], 3)

    def test_nest_metadata_records_anchor_and_mode(self):
        nested = self.composer.nest(self.alpha, self.beta, at="parse", mode="before")

        composition = nested.metadata["composition"]
        self.assertEqual(composition["operation"], "nest")
        self.assertEqual(composition["at"], "parse")
        self.assertEqual(composition["mode"], "before")

    def test_source_configs_merge_and_overrides_win(self):
        left = linear_pipeline(
            "left", [("a", lambda d, **k: d)], parallelism=2, retries=1
        )
        right = linear_pipeline("right", [("b", lambda d, **k: d)], parallelism=4)

        chained = self.composer.chain(left, right, config={"retries": 9})

        self.assertEqual(chained.config["parallelism"], 4)
        self.assertEqual(chained.config["retries"], 9)
        self.assertEqual(chained.metadata["parallelism"], 4)

    def test_source_config_is_not_mutated_by_the_merge(self):
        left = linear_pipeline("left", [("a", lambda d, **k: d)], parallelism=2)

        self.composer.chain(left, self.beta, config={"parallelism": 8})

        self.assertEqual(left.config["parallelism"], 2)

    def test_composed_pipeline_survives_a_serialization_round_trip(self):
        chained = self.composer.chain(self.alpha, self.beta, name="chained")
        serializer = PipelineSerializer()

        blob = serializer.serialize_pipeline(chained, format="json")
        restored = serializer.deserialize_pipeline(blob)

        self.assertEqual(
            [s.name for s in restored.steps],
            ["alpha.load", "alpha.parse", "beta.load"],
        )
        self.assertEqual(deps_by_name(restored)["beta.load"], ["alpha.parse"])

    def test_step_config_dependencies_track_the_rewired_dependencies(self):
        # PipelineBuilder.add_step() leaves a "dependencies" key in step.config,
        # which ExecutionEngine forwards to the handler as a kwarg. Rewiring must
        # not leave that copy pointing at pre-composition step names.
        builder = PipelineBuilder()
        builder.add_step("a", "math", handler=lambda d, **k: d)
        builder.add_step("b", "math", handler=lambda d, **k: d, dependencies=["a"])
        source = builder.build("source")

        chained = self.composer.chain(source)

        step_b = next(s for s in chained.steps if s.name == "source.b")
        self.assertEqual(step_b.dependencies, ["source.a"])
        self.assertEqual(step_b.config["dependencies"], ["source.a"])


class TestBuilderInclude(PipelineComposerTestCase):
    """PipelineBuilder.include() reuses a built pipeline as a sub-pipeline."""

    def test_included_steps_are_namespaced_and_wired_after_a_step(self):
        builder = PipelineBuilder()
        builder.add_step("seed", "math", handler=lambda data, **kwargs: data)
        builder.include(self.alpha, after="seed")
        built = builder.build("with_include")

        self.assertEqual(
            [s.name for s in built.steps], ["seed", "alpha.load", "alpha.parse"]
        )
        self.assertEqual(
            deps_by_name(built),
            {
                "seed": [],
                "alpha.load": ["seed"],
                "alpha.parse": ["alpha.load"],
            },
        )

    def test_returns_self_for_chaining(self):
        builder = PipelineBuilder()
        self.assertIs(builder.include(self.alpha), builder)

    def test_without_after_the_sub_pipeline_is_an_independent_branch(self):
        builder = PipelineBuilder()
        builder.add_step("seed", "math", handler=lambda data, **kwargs: data)
        builder.include(self.beta)
        built = builder.build("branches")

        self.assertEqual(deps_by_name(built)["beta.load"], [])

    def test_after_accepts_several_steps(self):
        builder = PipelineBuilder()
        builder.add_step("one", "math", handler=lambda data, **kwargs: data)
        builder.add_step("two", "math", handler=lambda data, **kwargs: data)
        builder.include(self.beta, after=["one", "two"])
        built = builder.build("fan_in")

        self.assertEqual(deps_by_name(built)["beta.load"], ["one", "two"])

    def test_the_same_pipeline_can_be_included_twice_under_prefixes(self):
        builder = PipelineBuilder()
        builder.include(self.beta, namespace="first")
        builder.include(self.beta, namespace="second")
        built = builder.build("twice")

        self.assertEqual([s.name for s in built.steps], ["first.load", "second.load"])

    def test_source_pipeline_is_not_mutated(self):
        builder = PipelineBuilder()
        builder.add_step("seed", "math", handler=lambda data, **kwargs: data)
        builder.include(self.alpha, after="seed")

        self.assertEqual([s.name for s in self.alpha.steps], ["load", "parse"])
        self.assertEqual(self.alpha.steps[0].dependencies, [])

    def test_duplicate_step_names_are_rejected(self):
        builder = PipelineBuilder()
        builder.add_step("load", "math", handler=lambda data, **kwargs: data)

        with self.assertRaises(ValidationError) as ctx:
            builder.include(self.beta, namespace=False)

        self.assertIn("already", str(ctx.exception))

    def test_unknown_after_step_is_rejected(self):
        builder = PipelineBuilder()
        builder.add_step("seed", "math", handler=lambda data, **kwargs: data)

        with self.assertRaises(ValidationError) as ctx:
            builder.include(self.alpha, after="nowhere")

        self.assertIn("not found", str(ctx.exception))

    def test_empty_pipeline_is_rejected(self):
        builder = PipelineBuilder()

        with self.assertRaises(ValidationError) as ctx:
            builder.include(Pipeline(name="empty"))

        self.assertIn("no steps", str(ctx.exception))

    def test_included_pipeline_executes(self):
        builder = PipelineBuilder()
        builder.add_step("seed", "math", handler=lambda data, **kwargs: data + 100)
        builder.include(self.alpha, after="seed")
        built = builder.build("with_include")

        result = ExecutionEngine().execute_pipeline(built, data=1)

        self.assertTrue(result.success)
        self.assertEqual(result.output, 204)  # ((1 + 100) + 1) * 2


if __name__ == "__main__":
    unittest.main()
