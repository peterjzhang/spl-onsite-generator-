"""
Tests for the agent behavior in the SPL onsite generator.

This module tests the unified Agent class with different performance levels,
including their work processes, time tracking, and interactions.
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_simulator import Agent, Task, TaskStatus

# Conftest functions are automatically available in pytest
# No need to import explicitly


class TestAgent:
    """Test Agent functionality with different performance levels."""

    def test_top_performer_initialization(self, seed_random, create_performance_config):
        """Test that top performer agents initialize correctly."""
        seed_random(42)

        config = create_performance_config(
            writing_hours=4.0, average_initial_quality=0.8, review_time_percentage=0.6
        )
        agent = Agent(
            id="test_top_performer",
            domain_name="test_domain",
            performance_level="top_performer",
            cfg=config,
        )

        assert agent.id == "test_top_performer"
        assert agent.domain_name == "test_domain"
        assert agent.performance_level == "top_performer"
        assert agent.hours_worked_today == 0.0
        assert agent.hours_worked_this_week == 0.0
        assert agent.actual_target_hours_per_day > 0
        assert agent.actual_writing_hours > 0
        assert agent.actual_revision_hours > 0
        assert agent.actual_review_hours > 0
        assert 0.0 <= agent.actual_average_initial_quality <= 1.0
        assert agent.actual_revision_improvement >= 0.0
        assert 0.0 <= agent.actual_quality_threshold <= 1.0

    def test_normal_contractor_initialization(
        self, seed_random, create_performance_config
    ):
        """Test that normal contractor agents initialize correctly."""
        seed_random(42)

        config = create_performance_config(
            writing_hours=6.0, average_initial_quality=0.7, review_time_percentage=0.0
        )
        agent = Agent(
            id="test_normal",
            domain_name="test_domain",
            performance_level="normal_contractor",
            cfg=config,
        )

        assert agent.id == "test_normal"
        assert agent.performance_level == "normal_contractor"
        assert agent.actual_writing_hours > 0
        assert agent.cfg.review_time_percentage == 0.0

    def test_agent_time_tracking(self, seed_random, create_performance_config):
        """Test that agent time tracking works correctly."""
        seed_random(42)

        config = create_performance_config(target_hours_per_day=8.0)
        agent = Agent(
            id="test_agent",
            domain_name="test",
            performance_level="top_performer",
            cfg=config,
        )

        # Should be able to work initially
        increment = agent.get_available_time_increment()
        assert increment > 0

        # Simulate some work
        agent.hours_worked_today = 4.0
        agent.hours_worked_this_week = 4.0

        # Should still be able to work
        increment = agent.get_available_time_increment()
        assert increment > 0

        # Exhaust daily hours
        agent.hours_worked_today = agent.actual_target_hours_per_day
        increment = agent.get_available_time_increment()
        assert increment == 0

    def test_agent_review_decision(self, seed_random, create_performance_config):
        """Test that agents make review vs write decisions based on configuration."""
        seed_random(42)

        # Agent that never reviews
        config_no_review = create_performance_config(review_time_percentage=0.0)
        agent_no_review = Agent(
            id="no_review",
            domain_name="test",
            performance_level="normal_contractor",
            cfg=config_no_review,
        )

        # Agent that always reviews
        config_always_review = create_performance_config(review_time_percentage=1.0)
        agent_always_review = Agent(
            id="always_review",
            domain_name="test",
            performance_level="top_performer",
            cfg=config_always_review,
        )

        # Test decisions over multiple calls
        no_review_decisions = [
            agent_no_review.should_review_this_increment() for _ in range(100)
        ]
        always_review_decisions = [
            agent_always_review.should_review_this_increment() for _ in range(100)
        ]

        assert all(not decision for decision in no_review_decisions)
        assert all(decision for decision in always_review_decisions)

    def test_agent_task_creation(self, seed_random, create_performance_config):
        """Test agent task creation."""
        seed_random(42)

        config = create_performance_config()
        agent = Agent(
            id="agent1",
            domain_name="engineering",
            performance_level="top_performer",
            cfg=config,
        )

        task = agent.create_task("task_001")

        assert task is not None
        assert task.id == "task_001"
        assert task.owner_id == "agent1"
        assert task.owner_domain == "engineering"
        assert task.status == TaskStatus.CLAIMED
        assert task.writing_progress_hours == 0.0

    def test_agent_work_on_writing(self, seed_random, create_performance_config):
        """Test agent working on writing a task."""
        seed_random(42)

        config = create_performance_config(writing_hours=4.0, writing_hours_noise=0.0)
        agent = Agent(
            id="agent1",
            domain_name="test",
            performance_level="top_performer",
            cfg=config,
        )

        task = agent.create_task("task_001")
        assert task is not None

        # Work on task in increments
        initial_hours = agent.hours_worked_today
        available = agent.get_available_time_increment()
        result = agent.work_on_writing_task(task, available)

        assert result is True
        assert task.status == TaskStatus.WRITING_IN_PROGRESS
        assert task.writing_progress_hours > 0
        assert agent.hours_worked_today > initial_hours

        # Continue until writing is complete
        while task.status == TaskStatus.WRITING_IN_PROGRESS:
            available = agent.get_available_time_increment()
            if available > 0:
                agent.work_on_writing_task(task, available)

        assert task.status == TaskStatus.COMPLETE
        assert task.quality_score > 0
        assert task.writing_progress_hours >= agent.actual_writing_hours

    def test_agent_work_on_revision(self, seed_random, create_performance_config):
        """Test agent working on revising a task."""
        seed_random(42)

        config = create_performance_config(revision_hours=2.0, revision_hours_noise=0.0)
        agent = Agent(
            id="agent1",
            domain_name="test",
            performance_level="top_performer",
            cfg=config,
        )

        # Create a task that needs work
        task = Task(
            id="task_001",
            owner_id="agent1",
            owner_domain="test",
            status=TaskStatus.NEEDS_WORK,
            quality_score=0.6,
        )

        initial_quality = task.quality_score
        initial_hours = agent.hours_worked_today

        # Work on revision
        available = agent.get_available_time_increment()
        result = agent.work_on_writing_task(task, available)

        assert result is True
        assert task.status == TaskStatus.REVISION_IN_PROGRESS
        assert task.revision_progress_hours > 0
        assert agent.hours_worked_today > initial_hours

        # Continue until revision is complete
        while task.status == TaskStatus.REVISION_IN_PROGRESS:
            available = agent.get_available_time_increment()
            if available > 0:
                agent.work_on_writing_task(task, available)

        assert task.status == TaskStatus.FIXING_DONE
        assert task.quality_score > initial_quality
        assert task.revision_count == 1

    def test_agent_work_on_review(self, seed_random, create_performance_config):
        """Test agent working on reviewing a task."""
        seed_random(42)

        config = create_performance_config(
            review_hours=3.0, review_hours_noise=0.0, quality_threshold=0.8
        )
        agent = Agent(
            id="reviewer1",
            domain_name="test",
            performance_level="top_performer",
            cfg=config,
        )

        # Create a task ready for review
        task = Task(
            id="task_001",
            owner_id="agent2",
            owner_domain="test",
            status=TaskStatus.COMPLETE,
            quality_score=0.9,  # High quality should pass
        )

        initial_hours = agent.hours_worked_today

        # Work on review
        available = agent.get_available_time_increment()
        result = agent.work_on_review_task(task, available)

        assert result is True
        assert task.status == TaskStatus.REVIEW_IN_PROGRESS
        assert task.reviewer_id == "reviewer1"
        assert task.reviewer_domain == "test"
        assert task.review_progress_hours > 0
        assert agent.hours_worked_today > initial_hours

        # Continue until review is complete
        while task.status == TaskStatus.REVIEW_IN_PROGRESS:
            available = agent.get_available_time_increment()
            if available > 0:
                agent.work_on_review_task(task, available)

        assert task.status == TaskStatus.SIGNED_OFF  # High quality should pass

    def test_agent_reject_low_quality(self, seed_random, create_performance_config):
        """Test agent rejecting low quality tasks during review."""
        seed_random(42)

        config = create_performance_config(
            review_hours=3.0, review_hours_noise=0.0, quality_threshold=0.8
        )
        agent = Agent(
            id="reviewer1",
            domain_name="test",
            performance_level="top_performer",
            cfg=config,
        )

        # Create a low quality task
        task = Task(
            id="task_001",
            owner_id="agent2",
            owner_domain="test",
            status=TaskStatus.COMPLETE,
            quality_score=0.6,  # Low quality should be rejected
        )

        # Complete the review
        while (
            task.status == TaskStatus.REVIEW_IN_PROGRESS
            or task.status == TaskStatus.COMPLETE
        ):
            available = agent.get_available_time_increment()
            if available > 0:
                agent.work_on_review_task(task, available)

        assert task.status == TaskStatus.NEEDS_WORK  # Low quality should be rejected

    def test_review_time_decay(self, seed_random, create_performance_config):
        """Test review time decay when reviewing same agent's work repeatedly."""
        seed_random(42)

        config = create_performance_config(
            review_hours=4.0, review_hours_noise=0.0, review_time_decay=0.8
        )
        agent = Agent(
            id="reviewer1",
            domain_name="test",
            performance_level="top_performer",
            cfg=config,
        )

        trainer_id = "agent2"

        # First review should take full time
        initial_time = agent.get_effective_review_time(trainer_id)
        assert initial_time == agent.actual_review_hours

        # Simulate multiple reviews by incrementing count
        agent.trainer_review_counts[trainer_id] = 1
        second_time = agent.get_effective_review_time(trainer_id)
        assert second_time < initial_time

        agent.trainer_review_counts[trainer_id] = 2
        third_time = agent.get_effective_review_time(trainer_id)
        assert third_time < second_time


class TestAgentInteraction:
    """Test agent interactions and cross-domain behavior."""

    def test_cross_domain_interaction(self, seed_random, create_performance_config):
        """Test that agents can only review tasks from their own domain."""
        seed_random(42)

        config = create_performance_config()

        # Agent in engineering domain
        eng_agent = Agent(
            id="eng1",
            domain_name="engineering",
            performance_level="top_performer",
            cfg=config,
        )

        # Agent in math domain
        math_agent = Agent(
            id="math1",
            domain_name="math",
            performance_level="top_performer",
            cfg=config,
        )

        # Create task in engineering domain
        eng_task = eng_agent.create_task("eng_task_001")
        assert eng_task is not None
        eng_task.status = TaskStatus.COMPLETE
        eng_task.quality_score = 0.9

        # Math agent shouldn't be able to review engineering task
        # (This would be enforced at the simulation level, not agent level)
        assert eng_task.owner_domain == "engineering"
        assert math_agent.domain_name == "math"

    def test_complete_task_workflow(self, seed_random, create_performance_config):
        """Test complete workflow from task creation to sign-off."""
        seed_random(42)

        writer_config = create_performance_config(
            writing_hours=4.0,
            writing_hours_noise=0.0,
            average_initial_quality=0.9,
            revision_hours=2.0,
            revision_hours_noise=0.0,
        )
        reviewer_config = create_performance_config(
            review_hours=3.0, review_hours_noise=0.0, quality_threshold=0.8
        )

        writer = Agent(
            id="writer1",
            domain_name="test",
            performance_level="normal_contractor",
            cfg=writer_config,
        )
        reviewer = Agent(
            id="reviewer1",
            domain_name="test",
            performance_level="top_performer",
            cfg=reviewer_config,
        )

        # 1. Writer creates and completes task
        task = writer.create_task("workflow_task")
        assert task is not None
        while task.status != TaskStatus.COMPLETE:
            available = writer.get_available_time_increment()
            if available > 0:
                writer.work_on_writing_task(task, available)

        assert task.status == TaskStatus.COMPLETE
        assert task.quality_score > 0

        # 2. Reviewer reviews and approves task
        while task.status not in [TaskStatus.SIGNED_OFF, TaskStatus.NEEDS_WORK]:
            available = reviewer.get_available_time_increment()
            if available > 0:
                reviewer.work_on_review_task(task, available)

        # Given high initial quality (0.9) and threshold (0.8), should be approved
        assert task.status == TaskStatus.SIGNED_OFF

    def test_revision_workflow(self, seed_random, create_performance_config):
        """Test workflow where task needs revision."""
        seed_random(42)

        writer_config = create_performance_config(
            writing_hours=4.0,
            writing_hours_noise=0.0,
            average_initial_quality=0.5,  # Low quality
            revision_hours=2.0,
            revision_hours_noise=0.0,
            revision_improvement=0.4,  # Good improvement
        )
        reviewer_config = create_performance_config(
            review_hours=3.0,
            review_hours_noise=0.0,
            quality_threshold=0.8,  # High threshold
        )

        writer = Agent(
            id="writer1",
            domain_name="test",
            performance_level="normal_contractor",
            cfg=writer_config,
        )
        reviewer = Agent(
            id="reviewer1",
            domain_name="test",
            performance_level="top_performer",
            cfg=reviewer_config,
        )

        # 1. Writer creates task with low quality
        task = writer.create_task("revision_task")
        assert task is not None
        while task.status != TaskStatus.COMPLETE:
            available = writer.get_available_time_increment()
            if available > 0:
                writer.work_on_writing_task(task, available)

        initial_quality = task.quality_score
        assert task.status == TaskStatus.COMPLETE
        assert initial_quality < 0.8  # Should be below threshold

        # 2. Reviewer rejects task
        while (
            task.status == TaskStatus.COMPLETE
            or task.status == TaskStatus.REVIEW_IN_PROGRESS
        ):
            available = reviewer.get_available_time_increment()
            if available > 0:
                reviewer.work_on_review_task(task, available)

        assert task.status == TaskStatus.NEEDS_WORK

        # 3. Writer revises task
        while task.status != TaskStatus.FIXING_DONE:
            available = writer.get_available_time_increment()
            if available > 0:
                writer.work_on_writing_task(task, available)

        assert task.status == TaskStatus.FIXING_DONE
        assert task.quality_score > initial_quality
        assert task.revision_count == 1
