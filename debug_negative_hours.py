#!/usr/bin/env python3
"""Debug script to understand negative hours issue."""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

from task_simulator import (
    Simulation,
    SimulationConfig,
    DomainSimulationSetup,
    TrainerConfig,
    ReviewerConfig,
    TaskStatus,
)


def debug_negative_hours():
    """Debug the negative hours issue by monitoring effective review times."""
    # Create the same config as the failing test
    trainer_cfg = TrainerConfig(
        max_hours_per_week=40.0,
        target_hours_per_day=6.0,
        target_hours_per_day_noise=0.0,
        writing_hours=4.0,
        writing_hours_noise=0.0,
        revision_hours=2.0,
        revision_hours_noise=0.0,
        average_initial_quality=0.7,
        average_initial_quality_noise=0.0,
        revision_improvement=0.1,
        revision_improvement_noise=0.0,
        revision_priority=0.7,
    )
    reviewer_cfg = ReviewerConfig(
        max_hours_per_week=40.0,
        target_hours_per_day=5.0,
        target_hours_per_day_noise=0.0,
        review_hours=3.0,
        review_hours_noise=0.0,
        review_time_decay=0.8,  # This causes effective time to decrease!
        quality_threshold=0.8,
        quality_threshold_noise=0.0,
    )

    domain = DomainSimulationSetup(
        domain_name="test",
        num_trainers=2,
        num_reviewers=2,
        trainer_cfg=trainer_cfg,
        reviewer_cfg=reviewer_cfg,
    )

    config = SimulationConfig(simulation_days=7, domain_setups=[domain], random_seed=42)

    # Create a custom simulation to access internal state
    class DebuggingSimulation(Simulation):
        def run(self):
            """Run simulation with detailed debugging."""
            self._seed_random_generators()
            self._initialize_agents()

            print(f"Initialized {len(self.reviewers)} reviewers:")
            for i, rev in enumerate(self.reviewers):
                print(
                    f"  Reviewer {i} ({rev.id}): target_hours={rev.actual_target_hours_per_day:.6f}"
                )
                print(f"    Base review hours: {rev.actual_review_hours:.6f}")
                reviewer_cfg: ReviewerConfig = rev.cfg  # type: ignore
                print(f"    Review time decay: {reviewer_cfg.review_time_decay:.6f}")

            daily_summary_data = []

            for self.day in range(1, self.config.simulation_days + 1):
                if (self.day - 1) % self.config.week_length_days == 0:
                    self.reset_agents_weekly_hours()

                print(f"\n=== DAY {self.day} ===")
                self.reset_agents_daily_hours()

                print("After daily reset:")
                for i, rev in enumerate(self.reviewers):
                    print(
                        f"  Reviewer {i}: hours_worked_today={rev.hours_worked_today:.6f}"
                    )

                increment_count = 0
                agents_still_working_in_day = True
                while agents_still_working_in_day:
                    increment_count += 1
                    agents_still_working_in_day = False

                    # Only track reviewer actions for brevity
                    for reviewer in self.reviewers:
                        available_increment = reviewer.get_available_time_increment()
                        if available_increment <= 0:
                            continue

                        # Check if reviewer has a current task
                        if reviewer.current_task_id and reviewer.current_phase:
                            current_task_obj = next(
                                (
                                    t
                                    for t in self.tasks
                                    if t.id == reviewer.current_task_id
                                ),
                                None,
                            )
                            if (
                                current_task_obj
                                and current_task_obj.status == reviewer.current_phase
                            ):
                                # Debug the effective review time calculation
                                trainer_id = current_task_obj.owner_id
                                if trainer_id:
                                    review_count = reviewer.trainer_review_counts.get(
                                        trainer_id, 0
                                    )
                                    effective_hours = (
                                        reviewer.get_effective_review_time(trainer_id)
                                    )
                                    needed = (
                                        effective_hours
                                        - current_task_obj.review_progress_hours
                                    )
                                    work_done = min(available_increment, needed)

                                    print(
                                        f"  Day {self.day}, Inc {increment_count}: {reviewer.id} reviewing {current_task_obj.id}"
                                    )
                                    print(
                                        f"    Trainer: {trainer_id}, Review count: {review_count}"
                                    )
                                    print(
                                        f"    Base review hours: {reviewer.actual_review_hours:.6f}"
                                    )
                                    reviewer_cfg: ReviewerConfig = reviewer.cfg  # type: ignore
                                    print(
                                        f"    Decay factor: {reviewer_cfg.review_time_decay ** review_count:.6f}"
                                    )
                                    print(f"    Effective hours: {effective_hours:.6f}")
                                    print(
                                        f"    Progress so far: {current_task_obj.review_progress_hours:.6f}"
                                    )
                                    print(f"    Hours needed: {needed:.6f}")
                                    print(
                                        f"    Available increment: {available_increment:.6f}"
                                    )
                                    print(f"    Work done: {work_done:.6f}")
                                    print(
                                        f"    Hours worked before: {reviewer.hours_worked_today:.6f}"
                                    )

                                if reviewer.work_on_review(
                                    current_task_obj, available_increment
                                ):
                                    agents_still_working_in_day = True
                                    print(
                                        f"    Hours worked after: {reviewer.hours_worked_today:.6f}"
                                    )

                                    if reviewer.hours_worked_today < 0:
                                        print(f"NEGATIVE HOURS DETECTED! {reviewer.id}")
                                        return pd.DataFrame()

                        # Also process new tasks (abbreviated logic)
                        tasks_for_reviewer = [
                            t
                            for t in self.tasks
                            if (
                                (
                                    t.status == TaskStatus.COMPLETE
                                    or t.status == TaskStatus.FIXING_DONE
                                )
                                and t.owner_domain == reviewer.domain_name
                            )
                        ]

                        if tasks_for_reviewer and not reviewer.current_task_id:
                            task_to_review = tasks_for_reviewer[0]
                            available_increment_for_review = (
                                reviewer.get_available_time_increment()
                            )
                            if available_increment_for_review > 0:
                                print(
                                    f"  Day {self.day}, Inc {increment_count}: {reviewer.id} starting new review {task_to_review.id}"
                                )
                                if reviewer.work_on_review(
                                    task_to_review, available_increment_for_review
                                ):
                                    agents_still_working_in_day = True
                                    if reviewer.hours_worked_today < 0:
                                        print(f"NEGATIVE HOURS DETECTED! {reviewer.id}")
                                        return pd.DataFrame()

                    # Process trainers too but without debug output
                    for trainer in self.trainers:
                        action_taken, _, _, _ = self._process_trainer_actions(
                            trainer, 0, 0, 0
                        )
                        if action_taken:
                            agents_still_working_in_day = True

                print(f"Day {self.day} end - Reviewer hours:")
                for i, rev in enumerate(self.reviewers):
                    print(f"  Reviewer {i}: {rev.hours_worked_today:.6f}")

                daily_summary_data.append(self._collect_daily_metrics(0, 0, 0, 0))

                avg_hours = np.mean([rev.hours_worked_today for rev in self.reviewers])
                if avg_hours < 0:
                    print("NEGATIVE AVERAGE DETECTED!")
                    break

            # Return summary
            df = pd.DataFrame(daily_summary_data)
            return df

    sim = DebuggingSimulation(config)
    results = sim.run()

    if len(results) > 0:
        print("\nFinal results:")
        print(results[["day", "avg_reviewer_hrs_worked_today"]].to_string())


if __name__ == "__main__":
    debug_negative_hours()
