"""
Tests for the agent behavior in the SPL onsite generator.

This module tests the TrainerAgent and ReviewerAgent classes,
including their work processes, time tracking, and interactions.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_simulator import TrainerAgent, ReviewerAgent, Task, TaskStatus

# Conftest functions are automatically available in pytest
# No need to import explicitly


class TestTrainerAgent:
    """Test TrainerAgent functionality."""

    def test_trainer_initialization(self, seed_random, create_trainer_config):
        """Test that trainer agents initialize correctly."""
        seed_random(42)

        config = create_trainer_config()
        trainer = TrainerAgent(id="test_trainer", domain_name="test_domain", cfg=config)

        assert trainer.id == "test_trainer"
        assert trainer.domain_name == "test_domain"
        assert trainer.hours_worked_today == 0.0
        assert trainer.hours_worked_this_week == 0.0
        assert trainer.actual_target_hours_per_day > 0
        assert trainer.actual_writing_hours > 0
        assert trainer.actual_revision_hours > 0
        assert 0.0 <= trainer.actual_average_initial_quality <= 1.0
        assert trainer.actual_revision_improvement >= 0.0

    def test_trainer_time_tracking(self, seed_random, create_trainer_config):
        """Test that trainer time tracking works correctly."""
        seed_random(42)

        config = create_trainer_config(target_hours_per_day=8.0)
        trainer = TrainerAgent(id="test_trainer", domain_name="test", cfg=config)

        # Should be able to work initially
        increment = trainer.get_available_time_increment()
        assert increment > 0

        # Simulate some work
        trainer.hours_worked_today = 4.0
        trainer.hours_worked_this_week = 4.0

        # Should still be able to work
        increment = trainer.get_available_time_increment()
        assert increment > 0

        # Exhaust daily hours
        trainer.hours_worked_today = trainer.actual_target_hours_per_day
        increment = trainer.get_available_time_increment()
        assert increment == 0

    def test_trainer_task_creation(self, seed_random, create_trainer_config):
        """Test trainer task creation."""
        seed_random(42)

        config = create_trainer_config()
        trainer = TrainerAgent(id="trainer1", domain_name="engineering", cfg=config)

        task = trainer.create_task("task_001")

        assert task is not None
        assert task.id == "task_001"
        assert task.owner_id == "trainer1"
        assert task.owner_domain == "engineering"
        assert task.status == TaskStatus.CLAIMED
        assert task.writing_progress_hours == 0.0

    def test_trainer_work_on_writing(self, seed_random, create_trainer_config):
        """Test trainer working on writing a task."""
        seed_random(42)

        config = create_trainer_config(writing_hours=4.0, writing_hours_noise=0.0)
        trainer = TrainerAgent(id="trainer1", domain_name="test", cfg=config)

        task = trainer.create_task("task_001")
        assert task is not None  # Type assertion for linter

        # Work on task in increments
        initial_hours = trainer.hours_worked_today
        available = trainer.get_available_time_increment()
        result = trainer.work_on_task(task, available)

        assert result is True
        assert task.status == TaskStatus.WRITING_IN_PROGRESS
        assert task.writing_progress_hours > 0
        assert trainer.hours_worked_today > initial_hours

        # Continue until writing is complete
        while task.status == TaskStatus.WRITING_IN_PROGRESS:
            available = trainer.get_available_time_increment()
            if available > 0:
                trainer.work_on_task(task, available)

        assert task.status == TaskStatus.COMPLETE
        assert task.quality_score > 0
        assert task.writing_progress_hours >= trainer.actual_writing_hours

    def test_trainer_work_on_revision(self, seed_random, create_trainer_config):
        """Test trainer working on revising a task."""
        seed_random(42)

        config = create_trainer_config(revision_hours=2.0, revision_hours_noise=0.0)
        trainer = TrainerAgent(id="trainer1", domain_name="test", cfg=config)

        # Create a task that needs work
        task = Task(
            id="task_001",
            owner_id="trainer1",
            owner_domain="test",
            status=TaskStatus.NEEDS_WORK,
            quality_score=0.6,
        )

        initial_quality = task.quality_score
        initial_hours = trainer.hours_worked_today

        # Work on revision
        available = trainer.get_available_time_increment()
        result = trainer.work_on_task(task, available)

        assert result is True
        assert task.status == TaskStatus.REVISION_IN_PROGRESS
        assert task.revision_progress_hours > 0
        assert trainer.hours_worked_today > initial_hours

        # Continue until revision is complete
        while task.status == TaskStatus.REVISION_IN_PROGRESS:
            available = trainer.get_available_time_increment()
            if available > 0:
                trainer.work_on_task(task, available)

        assert task.status == TaskStatus.FIXING_DONE
        assert task.quality_score > initial_quality
        assert task.revision_count == 1


class TestReviewerAgent:
    """Test ReviewerAgent functionality."""

    def test_reviewer_initialization(self, seed_random, create_reviewer_config):
        """Test that reviewer agents initialize correctly."""
        seed_random(42)

        config = create_reviewer_config()
        reviewer = ReviewerAgent(
            id="test_reviewer", domain_name="test_domain", cfg=config
        )

        assert reviewer.id == "test_reviewer"
        assert reviewer.domain_name == "test_domain"
        assert reviewer.hours_worked_today == 0.0
        assert reviewer.hours_worked_this_week == 0.0
        assert reviewer.actual_target_hours_per_day > 0
        assert reviewer.actual_review_hours > 0
        assert 0.0 <= reviewer.actual_quality_threshold <= 1.0
        assert len(reviewer.trainer_review_counts) == 0

    def test_reviewer_time_tracking(self, seed_random, create_reviewer_config):
        """Test that reviewer time tracking works correctly."""
        seed_random(42)

        config = create_reviewer_config(target_hours_per_day=6.0)
        reviewer = ReviewerAgent(id="test_reviewer", domain_name="test", cfg=config)

        # Should be able to work initially
        increment = reviewer.get_available_time_increment()
        assert increment > 0

        # Simulate some work
        reviewer.hours_worked_today = 3.0
        reviewer.hours_worked_this_week = 3.0

        # Should still be able to work
        increment = reviewer.get_available_time_increment()
        assert increment > 0

        # Exhaust daily hours
        reviewer.hours_worked_today = reviewer.actual_target_hours_per_day
        increment = reviewer.get_available_time_increment()
        assert increment == 0

    def test_reviewer_work_on_review(self, seed_random, create_reviewer_config):
        """Test reviewer working on reviewing a task."""
        seed_random(42)

        config = create_reviewer_config(
            review_hours=3.0, review_hours_noise=0.0, quality_threshold=0.8
        )
        reviewer = ReviewerAgent(id="reviewer1", domain_name="test", cfg=config)

        # Create a task ready for review
        task = Task(
            id="task_001",
            owner_id="trainer1",
            owner_domain="test",
            status=TaskStatus.COMPLETE,
            quality_score=0.9,  # High quality - should pass
        )

        initial_hours = reviewer.hours_worked_today

        # Work on review
        available = reviewer.get_available_time_increment()
        result = reviewer.work_on_review(task, available)

        assert result is True
        assert task.status == TaskStatus.REVIEW_IN_PROGRESS
        assert task.reviewer_id == "reviewer1"
        assert task.reviewer_domain == "test"
        assert task.review_progress_hours > 0
        assert reviewer.hours_worked_today > initial_hours

        # Continue until review is complete
        while task.status == TaskStatus.REVIEW_IN_PROGRESS:
            available = reviewer.get_available_time_increment()
            if available > 0:
                reviewer.work_on_review(task, available)

        assert task.status == TaskStatus.SIGNED_OFF  # High quality task
        # Check against effective review time, not base review time (accounts for decay)
        effective_review_time = reviewer.get_effective_review_time("trainer1")
        assert task.review_progress_hours >= effective_review_time

    def test_reviewer_reject_low_quality(self, seed_random, create_reviewer_config):
        """Test reviewer rejecting low quality tasks."""
        seed_random(42)

        config = create_reviewer_config(
            review_hours=2.0, review_hours_noise=0.0, quality_threshold=0.8
        )
        reviewer = ReviewerAgent(id="reviewer1", domain_name="test", cfg=config)

        # Create a low quality task
        task = Task(
            id="task_001",
            owner_id="trainer1",
            owner_domain="test",
            status=TaskStatus.COMPLETE,
            quality_score=0.5,  # Low quality - should be rejected
        )

        # Complete the review
        while (
            task.status == TaskStatus.COMPLETE
            or task.status == TaskStatus.REVIEW_IN_PROGRESS
        ):
            available = reviewer.get_available_time_increment()
            if available > 0:
                reviewer.work_on_review(task, available)
            else:
                break

        assert task.status == TaskStatus.NEEDS_WORK
        assert task.reviewer_id == "reviewer1"

    def test_review_time_decay(self, seed_random, create_reviewer_config):
        """Test that review time decreases with repeated reviews of same trainer."""
        seed_random(42)

        config = create_reviewer_config(
            review_hours=4.0, review_hours_noise=0.0, review_time_decay=0.8
        )
        reviewer = ReviewerAgent(id="reviewer1", domain_name="test", cfg=config)

        # First review should take full time
        first_time = reviewer.get_effective_review_time("trainer1")
        assert abs(first_time - 4.0) < 0.01

        # Simulate a completed review (increment counter)
        reviewer.trainer_review_counts["trainer1"] = 1

        # Second review should be faster (4.0 * 0.8 = 3.2)
        second_time = reviewer.get_effective_review_time("trainer1")
        assert abs(second_time - 3.2) < 0.01

        # Third review should be even faster (4.0 * 0.8^2 = 2.56)
        reviewer.trainer_review_counts["trainer1"] = 2
        third_time = reviewer.get_effective_review_time("trainer1")
        assert abs(third_time - 2.56) < 0.01


class TestAgentInteraction:
    """Test interactions between trainers and reviewers."""

    def test_cross_domain_interaction(
        self, seed_random, create_trainer_config, create_reviewer_config
    ):
        """Test that agents can work on tasks from their own domain."""
        seed_random(42)

        trainer_config = create_trainer_config()
        reviewer_config = create_reviewer_config()

        trainer = TrainerAgent(
            id="trainer1", domain_name="engineering", cfg=trainer_config
        )
        reviewer = ReviewerAgent(
            id="reviewer1", domain_name="engineering", cfg=reviewer_config
        )

        # Trainer creates task
        task = trainer.create_task("cross_domain_task")
        assert task is not None  # Type assertion for linter
        assert task.owner_domain == "engineering"

        # Complete writing
        while task.status != TaskStatus.COMPLETE:
            available = trainer.get_available_time_increment()
            if available > 0:
                trainer.work_on_task(task, available)
            else:
                break

        # Reviewer reviews task
        while task.status not in [TaskStatus.SIGNED_OFF, TaskStatus.NEEDS_WORK]:
            available = reviewer.get_available_time_increment()
            if available > 0:
                reviewer.work_on_review(task, available)
            else:
                break

        assert task.reviewer_domain == "engineering"
        assert task.status in [TaskStatus.SIGNED_OFF, TaskStatus.NEEDS_WORK]

    def test_complete_task_workflow(
        self, seed_random, create_trainer_config, create_reviewer_config
    ):
        """Test a complete task workflow from creation to sign-off."""
        seed_random(42)

        trainer_config = create_trainer_config(
            writing_hours=2.0,
            writing_hours_noise=0.0,
            revision_hours=1.0,
            revision_hours_noise=0.0,
            average_initial_quality=0.6,
        )
        reviewer_config = create_reviewer_config(
            review_hours=1.5, review_hours_noise=0.0, quality_threshold=0.7
        )

        trainer = TrainerAgent(id="trainer1", domain_name="test", cfg=trainer_config)
        reviewer = ReviewerAgent(
            id="reviewer1", domain_name="test", cfg=reviewer_config
        )

        # 1. Create task
        task = trainer.create_task("workflow_task")
        assert task is not None  # Type assertion for linter
        assert task.status == TaskStatus.CLAIMED

        # 2. Complete writing
        while task.status != TaskStatus.COMPLETE:
            available = trainer.get_available_time_increment()
            if available > 0:
                trainer.work_on_task(task, available)
            else:
                break

        assert task.status == TaskStatus.COMPLETE
        original_quality = task.quality_score

        # 3. First review (might reject due to quality)
        while task.status not in [TaskStatus.SIGNED_OFF, TaskStatus.NEEDS_WORK]:
            available = reviewer.get_available_time_increment()
            if available > 0:
                reviewer.work_on_review(task, available)
            else:
                break

        if task.status == TaskStatus.NEEDS_WORK:
            # 4. Revision needed
            while task.status != TaskStatus.FIXING_DONE:
                available = trainer.get_available_time_increment()
                if available > 0:
                    trainer.work_on_task(task, available)
                else:
                    break

            assert task.status == TaskStatus.FIXING_DONE
            assert task.quality_score > original_quality

            # 5. Second review
            while task.status not in [TaskStatus.SIGNED_OFF, TaskStatus.NEEDS_WORK]:
                available = reviewer.get_available_time_increment()
                if available > 0:
                    reviewer.work_on_review(task, available)
                else:
                    break

        # Task should eventually be signed off or need more work
        assert task.status in [TaskStatus.SIGNED_OFF, TaskStatus.NEEDS_WORK]
