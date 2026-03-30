from steptronoss.core.generators.flow_controller import FlowController, SimpleFlowController
from steptronoss.core.generators.flow_controller_simulator import (
    FlowControllerSimulator,
    FlowControllerSimulatorCLI,
    FlowSimulationResult,
    SimulatedFlowDataloader,
    _normalize_genables,
    simulate_flow_controller,
)
from steptronoss.exp.inference import VLLMDeployConfig
from steptronoss.exp.rl import FlowControllerConfig, FullyAsyncFlowControllerConfig


def build_controller(
    strategy: str,
    prompt_per_iter: int = 3,
) -> tuple[FlowControllerConfig | FullyAsyncFlowControllerConfig, FlowController]:
    if strategy == "fully-async":
        cfg = FullyAsyncFlowControllerConfig()
        cfg.prompt_per_iter = prompt_per_iter
        cfg.max_untrained_prompts = 4
        cfg.max_staleness = 2
        cfg.vllm_cfg = VLLMDeployConfig()
        return cfg, cfg.build_flow_controller()
    cfg = FlowControllerConfig()
    cfg.async_strategy = strategy
    cfg.prompt_per_iter = prompt_per_iter
    return cfg, SimpleFlowController(flow_cfg=cfg)


def test_one_step_off_matches_expected_ascii_timeline():
    cfg, controller = build_controller("one-step-off")

    result = simulate_flow_controller(
        cfg,
        infer_costs=[1, 2, 3, 1, 2, 3],
        train_cost=4,
    )

    assert [block.required_version for block in result.blocks] == [0, 0]
    assert [snapshot.entries for snapshot in result.yield_snapshots] == [(), ()]
    assert [snapshot.yielded_ids for snapshot in result.yield_snapshots] == [(0, 1, 2), (3, 4, 5)]
    assert result.train_timeline == ["N", "N", "N", "3", "T", "T", "T", "T", "3", "T", "T", "T", "T"]
    assert result.infer_timeline == ["3", "2", "1", "Y", "3", "2", "1", "N", "Y", "N", "N", "N", "N"]


def test_on_policy_waits_for_next_weight_before_next_block():
    cfg, controller = build_controller("on-policy")

    result = simulate_flow_controller(
        cfg,
        infer_costs=[1, 2, 3, 1, 2, 3],
        train_cost=4,
    )

    assert [block.required_version for block in result.blocks] == [0, 1]
    assert result.blocks[1].sync_time == 7
    assert result.blocks[1].gen_start == 7
    assert result.train_timeline == [
        "N",
        "N",
        "N",
        "3",
        "T",
        "T",
        "T",
        "T",
        "N",
        "N",
        "N",
        "3",
        "T",
        "T",
        "T",
        "T",
    ]
    assert result.infer_timeline == [
        "3",
        "2",
        "1",
        "Y",
        "N",
        "N",
        "N",
        "N",
        "3",
        "2",
        "1",
        "Y",
        "N",
        "N",
        "N",
        "N",
    ]


def test_simple_strategies_honor_explicit_max_concurrent_limit():
    cfg, controller = build_controller("one-step-off", prompt_per_iter=4)

    result = simulate_flow_controller(
        cfg,
        infer_costs=[1, 2, 3, 4],
        train_cost=0,
        max_concurrent=2,
    )

    assert result.blocks[0].infer_concurrency == (2, 2, 2, 2, 1, 1)
    assert result.blocks[0].ready_time == 6
    assert result.infer_timeline == ["2", "2", "2", "2", "1", "1", "Y"]


def test_simulated_dataloader_is_restored_after_simulation():
    cfg, controller = build_controller("one-step-off", prompt_per_iter=2)
    dataloader = SimulatedFlowDataloader([2, 1, 2])
    simulator = FlowControllerSimulator(cfg)

    result = simulator.simulate(dataloader=dataloader, train_cost=1)

    assert result.genable_costs == [2, 1, 2]
    assert dataloader.state_dict() == {"index": 0}


def test_fire_helpers_accept_csv_and_list_inputs():
    assert _normalize_genables("1, 2 3") == [1, 2, 3]
    assert _normalize_genables([1, "2", 3]) == [1, 2, 3]
    single_render = FlowControllerSimulatorCLI().render(
        strategy="one-step-off",
        prompt_per_iter=1,
        train_cost=0,
        genables="0",
    )
    assert "Train: 1" in single_render

    rendered = FlowControllerSimulatorCLI().render(
        strategy="one-step-off",
        prompt_per_iter=3,
        train_cost=4,
        genables="1,2,3,1,2,3",
    )

    assert "Genables: [1, 2, 3, 1, 2, 3] TrainCost: 4" in rendered
    assert "Legend: T: Training Y: Yield InferDigits: Concurrent Gen" in rendered
    assert "N: Idle" not in rendered
    expected_train = "Train: " + FlowSimulationResult._render_timeline([
        "N",
        "N",
        "N",
        "3",
        "T",
        "T",
        "T",
        "T",
        "3",
        "T",
        "T",
        "T",
        "T",
    ])
    assert expected_train in rendered


def test_fully_async_waits_for_prompt_per_iter_before_training():
    cfg, controller = build_controller("fully-async", prompt_per_iter=2)
    cfg.max_untrained_prompts = 8
    cfg.max_staleness = 8

    result = simulate_flow_controller(
        cfg,
        infer_costs=[3, 3, 3, 3],
        train_cost=1,
        max_concurrent=1,
    )

    assert [(batch.batch_index, batch.start_time, batch.prompt_ids) for batch in result.train_batches] == [
        (0, 6, (0, 1)),
        (1, 12, (2, 3)),
    ]
    assert result.yield_snapshots[0].entries == ()
    assert result.yield_snapshots[1].entries == ()
    assert result.yield_snapshots[0].yielded_ids == (0, 1)
    assert result.yield_snapshots[1].yielded_ids == (2, 3)
    assert result.prompt_traces[0].completion_time == 3
    assert result.prompt_traces[1].completion_time == 6


def test_fully_async_backpressure_limits_untrained_prompts():
    cfg, controller = build_controller("fully-async", prompt_per_iter=2)
    cfg.max_untrained_prompts = 2
    cfg.max_staleness = 8

    result = simulate_flow_controller(
        cfg,
        infer_costs=[1, 1, 1, 1],
        train_cost=4,
        max_concurrent=1,
    )

    launch_times = {trace.item_id: trace.launch_time for trace in result.prompt_traces}
    assert launch_times[0] == 0
    assert launch_times[1] == 1
    assert launch_times[2] >= result.train_batches[0].start_time
    assert launch_times[3] >= result.train_batches[0].start_time


def test_fully_async_max_concurrent_advances_all_running_prompts_each_tick():
    cfg, controller = build_controller("fully-async", prompt_per_iter=2)
    cfg.max_untrained_prompts = 8
    cfg.max_staleness = 8

    result = simulate_flow_controller(
        cfg,
        infer_costs=[3, 3],
        train_cost=0,
        max_concurrent=2,
    )

    assert [trace.completion_time for trace in result.prompt_traces] == [3, 3]
    assert result.train_batches[0].start_time == 3
    assert result.infer_timeline == ["2", "2", "2", "Y"]


def test_fully_async_staleness_blocks_next_update_until_long_tail_finishes():
    cfg, controller = build_controller("fully-async", prompt_per_iter=1)
    cfg.max_untrained_prompts = 8
    cfg.max_staleness = 1

    result = simulate_flow_controller(
        cfg,
        infer_costs=[5, 1, 1],
        train_cost=1,
        max_concurrent=2,
    )

    assert [batch.train_version for batch in result.train_batches] == [0, 1, 2]
    assert result.train_batches[1].start_time < result.prompt_traces[0].completion_time
    assert result.train_batches[2].start_time >= result.prompt_traces[0].completion_time
    rendered = result.render()
    assert "Legend: T: Training Y: Yield InferDigits: Concurrent Gen" in rendered
    assert "<U" not in rendered
    assert "<W" not in rendered
    assert "Yield 0: trunc[0(1/5)] yield[1]" in rendered
    assert "Yield 1: trunc[0(2/5)] yield[2]" in rendered


def test_fully_async_rejects_deadlocking_config():
    cfg, controller = build_controller("fully-async", prompt_per_iter=4)
    cfg.max_untrained_prompts = 2

    try:
        simulate_flow_controller(
            cfg,
            infer_costs=[1, 2, 3, 4, 1, 2, 3, 4],
            train_cost=3,
            max_concurrent=1,
        )
    except ValueError as exc:
        assert "deadlocks" in str(exc)
    else:
        raise AssertionError("Expected fully-async invalid config to fail fast")


def test_fully_async_refills_before_first_yield_when_equal_to_backpressure_limit():
    cfg, controller = build_controller("fully-async", prompt_per_iter=4)
    cfg.max_untrained_prompts = 4
    cfg.max_staleness = 2

    result = simulate_flow_controller(
        cfg,
        infer_costs=[1, 2, 3, 4, 1, 2, 3, 4],
        train_cost=3,
        max_concurrent=2,
    )

    first_yield = result.infer_timeline.index("Y")
    assert set(result.infer_timeline[:first_yield]) == {"2"}
