import json
import traceback
from pathlib import Path

import py_trees

from leaderboard.state_estimation.state_from_sensors import CarlaStateEstimatorEngine
from leaderboard.state_estimation.atomic_criteria_sensors import (
    RouteCompletionTest, OutsideRouteLanesTest, InRouteTest, CollisionTest,
    RunningRedLightTest, RunningStopTest, MinimumSpeedRouteTest, ActorBlockedTest,
    OnSidewalkTest, WrongLaneTest)


class ShadowCriteria:
    MAX_ERRORS = 5

    def __init__(self, ego, route):
        self.ego = ego
        self.route = route
        self.criteria = []
        self.errors = {}
        self._built = False
        self._finalized = False
        self._disabled = set()
        self.build_failed = []
        self.update_counts = {}
        self.n_ticks = 0            # ticks where update() was called
        self.n_ticks_no_state = 0   # ticks skipped because the estimator had no state

    def _specs(self):
        a, r = self.ego, self.route
        return [
            ("OnSidewalkTest", lambda: OnSidewalkTest(a)),
            ("WrongLaneTest", lambda: WrongLaneTest(a)),
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
                self.update_counts[c.name] = 0
            except Exception:
                self._note_error(name)
                self._disabled.add(name)
                self.build_failed.append(name)
        self._built = True

    def update(self):
        """Call once per tick, after the estimator has been updated. Never raises."""
        try:
            self.n_ticks += 1
            if CarlaStateEstimatorEngine.get_calculated_state() is None:
                self.n_ticks_no_state += 1
                return
            if not self._built:
                self._build()
            for c in self.criteria:
                if c.name in self._disabled:
                    continue
                try:
                    c.update()
                    self.update_counts[c.name] += 1
                except Exception:
                    self._note_error(c.name)
        except Exception:
            self._note_error("__runner__")

    def finalize(self):
        if self._finalized:
            return
        self._finalized = True
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
                    events.append({"type": str(ev.get_type()), "frame": ev.get_frame(),
                                   "message": ev.get_message()})
                except Exception:
                    events.append({"type": "?", "frame": None, "message": "<unreadable event>"})
            out.append({"name": c.name, "status": c.test_status,
                        "n_updates": self.update_counts.get(c.name, 0),
                        "actual_value": c.actual_value, "success_value": c.success_value,
                        "units": c.units, "num_events": len(events), "events": events,
                        "disabled_after_errors": c.name in self._disabled})
        return {"n_ticks": self.n_ticks,
                "n_ticks_no_state": self.n_ticks_no_state,
                "build_failed": self.build_failed,
                "criteria": out,
                "errors": {k: {"count": v["count"], "first_traceback": v["first"]}
                           for k, v in self.errors.items()}}

    def save_json(self, path):
        self.finalize()
        Path(path).write_text(json.dumps(self.results(), indent=2, default=str))