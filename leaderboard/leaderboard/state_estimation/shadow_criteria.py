"""
Runs the state-estimator-based criteria NEXT TO the official Bench2Drive criteria tree ("shadow mode").

* The official py_trees criteria tree is not touched: B2D scoring, termination and results stay exactly as they were.
* Your criteria are updated once per tick by hand, never raise into the scenario manager, and write their
  own JSON at the end of each route.
* Criteria are built lazily on the first tick that has a calculated state, because several of them read
  CarlaStateEstimatorEngine.get_calculated_state() already inside __init__/initialise().
"""
import json
import traceback
from pathlib import Path

import py_trees

# IMPORTANT: use exactly the same import path as scenario_manager.py uses for the engine; a different path
# (e.g. leaderboard.leaderboard.state_estimation...) creates a second copy of the class with its own empty state.
from leaderboard.state_estimation.state_from_sensors import CarlaStateEstimatorEngine
from leaderboard.state_estimation.atomic_criteria_sensors import (
RouteCompletionTest, OutsideRouteLanesTest, InRouteTest, CollisionTest,
    RunningRedLightTest, RunningStopTest, MinimumSpeedRouteTest, ActorBlockedTest,
    MaxVelocityTest, OnSidewalkTest, WrongLaneTest)


class ShadowCriteria:
    MAX_ERRORS = 5          

    def __init__(self, ego, route):
        self.ego = ego
        self.route = route                    # scenario.route: list of (carla.Transform, RoadOption)RouteScenario
        self.criteria = []
        self.errors = {}                      # criterion name -> {"count": n, "first": traceback}
        self._built = False
        self._disabled = set()

    def _specs(self):
        a, r = self.ego, self.route
        return [
            ("MaxVelocityTest", lambda: MaxVelocityTest(a, route=r)),
            ("OnSidewalkTest", lambda: OnSidewalkTest(a, route=r)),
            ("WrongLaneTest", lambda: WrongLaneTest(a, route=r)),
            ("RouteCompletionTest", lambda: RouteCompletionTest(a, route=r)),
            ("OutsideRouteLanesTest", lambda: OutsideRouteLanesTest(a, route=r)),
            ("InRouteTest", lambda: InRouteTest(a, route=r, offroad_max=30, terminate_on_failure=False)),
            ("CollisionTest", lambda: CollisionTest(a, name="CollisionTest")),
            ("RunningRedLightTest", lambda: RunningRedLightTest(a)),
            ("RunningStopTest", lambda: RunningStopTest(a)),
            ("MinSpeedTest", lambda: MinimumSpeedRouteTest(a, route=r, checkpoints=4, name="MinSpeedTest")),
            ("AgentBlockedTest", lambda: ActorBlockedTest(a, min_speed=0.1, max_time=60.0,
                                                         terminate_on_failure=False, name="AgentBlockedTest")),
        ]

    def _note_error(self, name):
        e = self.errors.setdefault(name, {"count": 0, "first": traceback.format_exc()})
        e["count"] += 1
        if e["count"] >= self.MAX_ERRORS:
            self._disabled.add(name)

    def _build(self):
        for name, make in self._specs():
            try:
                c = make()
                c.initialise()
                self.criteria.append(c)
            except Exception:
                self._note_error(name)
                self._disabled.add(name)
        self._built = True

    # ---- called by the scenario manager ---------------------------------------------------------------
    def update(self):
        """Call once per tick, after the estimator has been updated. Never raises."""
        try:
            if CarlaStateEstimatorEngine.get_calculated_state() is None:
                return
            if not self._built:
                self._build()
            for c in self.criteria:
                if c.name in self._disabled:
                    continue
                try:
                    c.update()
                except Exception:
                    self._note_error(c.name)
        except Exception:
            self._note_error("__runner__")

    def finalize(self):
        for c in self.criteria:
            try:
                c.terminate(py_trees.common.Status.INVALID)
            except Exception:
                self._note_error(c.name)

    def results(self):
        out = []
        for c in self.criteria:
            events = []
            for ev in c.events:
                try:
                    events.append({"type": str(ev.get_type()), "frame": ev.get_frame(), "message": ev.get_message()})
                except Exception:
                    events.append({"type": "?", "frame": None, "message": "<unreadable event>"})
            out.append({"name": c.name, "status": c.test_status, "actual_value": c.actual_value,
                        "success_value": c.success_value, "units": c.units, "num_events": len(events),
                        "events": events, "disabled_after_errors": c.name in self._disabled})
        return {"criteria": out,
                "errors": {k: {"count": v["count"], "first_traceback": v["first"]} for k, v in self.errors.items()}}

    def save_json(self, path):
        Path(path).write_text(json.dumps(self.results(), indent=2, default=str))