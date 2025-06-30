from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
import pandas as pd
from typing import List, Optional

# import ace_tools as tools # Commented out as it's used for demo, not core sim logic

WORK_INCREMENT_HOURS = 0.5  # Minimum work block


def sample_gamma_from_mean_cv(mean: float, cv: float) -> float:
    """
    Sample from Gamma(shape, scale) where shape = 1/cv^2 and scale = mean*cv^2

    Args:
        mean: The desired mean of the distribution
        cv: The coefficient of variation (std/mean)

    Returns:
        A sample from the gamma distribution
    """
    # For Gamma distribution: mean = shape * scale, variance = shape * scale^2
    # We want: mean = given, cv = std/mean, so std = cv * mean
    # variance = (cv * mean)^2 = cv^2 * mean^2
    # Therefore: shape * scale^2 = cv^2 * mean^2
    # And: shape * scale = mean
    # Solving: shape = 1/cv^2, scale = mean * cv^2

    # Handle edge cases
    if mean <= 0 or cv <= 0:
        return max(0.01, mean)  # Fallback for invalid parameters

    shape = 1.0 / (cv * cv)
    scale = mean * cv * cv

    return np.random.gamma(shape, scale)


class TaskStatus(Enum):
    """Represents the various states a task can be in during its lifecycle."""

    CLAIMED = auto()  # Task created, not yet worked on (for writing)
    WRITING_IN_PROGRESS = auto()  # Writing is partially done
    COMPLETE = auto()  # Ready for first review (writing done)
    REVIEW = auto()  # Actively being reviewed (deprecated, use REVIEW_IN_PROGRESS)
    REVIEW_IN_PROGRESS = auto()  # Review is partially done
    NEEDS_WORK = auto()  # Reviewer sent back for revision
    REVISION_IN_PROGRESS = auto()  # Revision is partially done
    FIXING_DONE = auto()  # Ready for subsequent review (revision done)
    SIGNED_OFF = auto()


@dataclass
class PerformanceLevelConfig:
    """Configuration settings for agents at a specific performance level.

    Attributes:
        max_hours_per_week: Maximum hours an agent can work in a week.
        target_hours_per_day: Mean target daily work hours (Gamma distribution mean).
        target_hours_per_day_noise: Coefficient of variation for daily target hours (std/mean).
        writing_hours: Mean hours required to write a new task (Gamma distribution mean).
        writing_hours_noise: Coefficient of variation for task writing hours (std/mean).
        revision_hours: Mean hours required to revise a task (Gamma distribution mean).
        revision_hours_noise: Coefficient of variation for task revision hours (std/mean).
        review_hours: Mean hours required to review a task (Gamma distribution mean).
        review_hours_noise: Coefficient of variation for task review hours (std/mean).
        average_initial_quality: Average quality score of initially created tasks (Normal mean).
        average_initial_quality_noise: Standard deviation for initial quality (Normal stddev).
        revision_improvement: Mean improvement in quality score after a revision (Gamma distribution mean).
        revision_improvement_noise: Coefficient of variation for revision improvement (std/mean).
        quality_threshold: Minimum quality score for a task to be signed off (Normal mean).
        quality_threshold_noise: Standard deviation for quality threshold (Normal stddev).
        review_time_decay: Decay factor for review time when reviewing the same person repeatedly.
        revision_priority: Probability (0.0 to 1.0) an agent will prioritize revising over creating new tasks.
        review_time_percentage: Percentage of time (0.0 to 1.0) this performance level spends on reviewing.
    """

    max_hours_per_week: float = 40.0
    target_hours_per_day: float = 8.0
    target_hours_per_day_noise: float = 0.25
    writing_hours: float = 6.0
    writing_hours_noise: float = 0.3
    revision_hours: float = 1.5
    revision_hours_noise: float = 0.4
    review_hours: float = 2.0
    review_hours_noise: float = 0.35
    average_initial_quality: float = 0.7
    average_initial_quality_noise: float = 0.1
    revision_improvement: float = 0.1
    revision_improvement_noise: float = 0.5
    quality_threshold: float = 0.8
    quality_threshold_noise: float = 0.05
    review_time_decay: float = 0.9
    revision_priority: float = 0.7
    review_time_percentage: float = 0.3


@dataclass
class DomainSimulationSetup:
    """Configuration for a specific domain within a multi-domain simulation.

    Attributes:
        domain_name: The name of the domain.
        num_top_performers: Number of top performer agents in this domain.
        num_normal_contractors: Number of normal contractor agents in this domain.
        num_bad_contractors: Number of bad contractor agents in this domain.
        top_performer_cfg: Configuration object for top performers in this domain.
        normal_contractor_cfg: Configuration object for normal contractors in this domain.
        bad_contractor_cfg: Configuration object for bad contractors in this domain.
    """

    domain_name: str
    num_top_performers: int
    num_normal_contractors: int
    num_bad_contractors: int
    top_performer_cfg: PerformanceLevelConfig
    normal_contractor_cfg: PerformanceLevelConfig
    bad_contractor_cfg: PerformanceLevelConfig


@dataclass
class SimulationConfig:
    """Overall configuration for the simulation.

    Attributes:
        simulation_days: Total number of days the simulation will run.
        week_length_days: Number of days in a work week (for resetting weekly hours).
        domain_setups: A list of DomainSimulationSetup objects, defining each
                       domain to be simulated.
        random_seed: Optional random seed for deterministic simulation runs.
    """

    simulation_days: int = 21
    week_length_days: int = 7
    domain_setups: List[DomainSimulationSetup] = field(default_factory=list)
    random_seed: Optional[int] = None


@dataclass
class Task:
    """Represents a single task within the simulation.

    Attributes:
        id: Unique identifier for the task.
        owner_id: ID of the Agent who created the task.
        owner_domain: Domain of the agent who created the task.
        reviewer_id: ID of the Agent who reviewed/is reviewing the task.
        reviewer_domain: Domain of the reviewer.
        status: Current status of the task (TaskStatus enum).
        revision_count: Number of times the task has been revised.
        quality_score: Current quality score of the task (0.0 to 1.0).
        minor_issues: Estimated number of minor issues based on quality.
        major_issues: Estimated number of major issues based on quality.
        writing_progress_hours: Hours spent on initial writing.
        revision_progress_hours: Hours spent on revisions.
        review_progress_hours: Hours spent on reviews.
    """

    id: str
    owner_id: str
    owner_domain: Optional[str] = None
    reviewer_id: Optional[str] = None
    reviewer_domain: Optional[str] = None
    status: TaskStatus = TaskStatus.CLAIMED
    revision_count: int = 0
    quality_score: float = 0.0
    minor_issues: int = 0
    major_issues: int = 0
    # Progress tracking
    writing_progress_hours: float = 0.0
    revision_progress_hours: float = 0.0
    review_progress_hours: float = 0.0

    def update_from_quality(self):
        """Updates minor and major issue counts based on the current quality_score."""
        self.minor_issues = max(0, int((1 - self.quality_score) * 10))
        self.major_issues = max(0, int((1 - self.quality_score) * 5))


@dataclass
class Agent:
    """Represents an agent in the simulation who can both write and review tasks.

    Attributes:
        id: Unique identifier for the agent.
        domain_name: The domain this agent belongs to.
        performance_level: The performance level (top_performer, normal_contractor, bad_contractor).
        cfg: Configuration object for this performance level.
        actual_target_hours_per_day: The agent's specific target hours for the day.
        actual_writing_hours: Actual hours this agent takes to write a task.
        actual_revision_hours: Actual hours this agent takes to revise a task.
        actual_review_hours: Actual hours this agent takes to review a task.
        actual_average_initial_quality: Actual initial quality of tasks by this agent.
        actual_revision_improvement: Actual quality improvement per revision by this agent.
        actual_quality_threshold: Actual quality threshold this agent uses for sign-off.
        hours_worked_this_week: Total hours worked by the agent in the current week.
        hours_worked_today: Total hours worked by the agent today.
        current_task_id: ID of the task the agent is currently focused on.
        current_phase: The specific phase of work the agent is performing.
        trainer_review_counts: Dictionary tracking review counts for review time decay.
    """

    id: str
    domain_name: str
    performance_level: str
    cfg: PerformanceLevelConfig
    actual_target_hours_per_day: float = field(init=False)
    actual_writing_hours: float = field(init=False)
    actual_revision_hours: float = field(init=False)
    actual_review_hours: float = field(init=False)
    actual_average_initial_quality: float = field(init=False)
    actual_revision_improvement: float = field(init=False)
    actual_quality_threshold: float = field(init=False)
    hours_worked_this_week: float = 0.0
    hours_worked_today: float = 0.0
    current_task_id: Optional[str] = None
    current_phase: Optional[TaskStatus] = None
    trainer_review_counts: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Initializes actual agent parameters based on config noise."""
        # Sample target hours per day using Gamma distribution
        self.actual_target_hours_per_day = sample_gamma_from_mean_cv(
            self.cfg.target_hours_per_day, self.cfg.target_hours_per_day_noise
        )
        self.actual_target_hours_per_day = max(0.1, self.actual_target_hours_per_day)

        # Sample writing hours
        self.actual_writing_hours = sample_gamma_from_mean_cv(
            self.cfg.writing_hours, self.cfg.writing_hours_noise
        )
        self.actual_writing_hours = max(0.1, self.actual_writing_hours)

        # Sample revision hours
        self.actual_revision_hours = sample_gamma_from_mean_cv(
            self.cfg.revision_hours, self.cfg.revision_hours_noise
        )
        self.actual_revision_hours = max(0.1, self.actual_revision_hours)

        # Sample review hours
        self.actual_review_hours = sample_gamma_from_mean_cv(
            self.cfg.review_hours, self.cfg.review_hours_noise
        )
        self.actual_review_hours = max(0.1, self.actual_review_hours)

        # Sample initial quality (Normal distribution)
        self.actual_average_initial_quality = np.clip(
            np.random.normal(
                self.cfg.average_initial_quality,
                self.cfg.average_initial_quality_noise,
            ),
            0.0,
            1.0,
        )

        # Sample revision improvement
        if self.cfg.revision_improvement > 0:
            self.actual_revision_improvement = sample_gamma_from_mean_cv(
                self.cfg.revision_improvement, self.cfg.revision_improvement_noise
            )
        else:
            self.actual_revision_improvement = 0.0
        self.actual_revision_improvement = max(0.0, self.actual_revision_improvement)

        # Sample quality threshold (Normal distribution)
        self.actual_quality_threshold = np.clip(
            np.random.normal(
                self.cfg.quality_threshold, self.cfg.quality_threshold_noise
            ),
            0.0,
            1.0,
        )

        # Initialize trainer review counts
        self.trainer_review_counts = {}

    def reset_weekly_hours(self):
        """Resets the agent's weekly worked hours count."""
        self.hours_worked_this_week = 0.0

    def reset_daily_hours(self):
        """Resets the agent's daily worked hours count."""
        self.hours_worked_today = 0.0

    def get_available_time_increment(self) -> float:
        """Calculates the available time for the agent to work in the next increment."""
        remaining_today = self.actual_target_hours_per_day - self.hours_worked_today
        remaining_week = self.cfg.max_hours_per_week - self.hours_worked_this_week

        if remaining_today <= 0 or remaining_week <= 0:
            return 0.0

        return min(WORK_INCREMENT_HOURS, remaining_today, remaining_week)

    def should_review_this_increment(self) -> bool:
        """Decides whether this agent should spend this increment on reviewing vs writing."""
        return np.random.random() < self.cfg.review_time_percentage

    def get_effective_review_time(self, trainer_id: str) -> float:
        """Calculate the effective review time for a trainer's work, applying decay."""
        review_count = self.trainer_review_counts.get(trainer_id, 0)
        decay_multiplier = self.cfg.review_time_decay**review_count
        effective_time = self.actual_review_hours * decay_multiplier
        return max(0.1, effective_time)

    def create_task(self, task_id: str) -> Optional[Task]:
        """Creates a new Task object, assigning self as owner."""
        task = Task(
            id=task_id,
            owner_id=self.id,
            owner_domain=self.domain_name,
            status=TaskStatus.CLAIMED,
        )
        return task

    def work_on_writing_task(self, task: Task, available_increment: float) -> bool:
        """Performs writing work on a task (initial writing or revision)."""
        if task.status in (TaskStatus.CLAIMED, TaskStatus.WRITING_IN_PROGRESS):
            # Initial writing
            needed = self.actual_writing_hours - task.writing_progress_hours
            work_done = min(available_increment, needed)
            task.writing_progress_hours += work_done
            self.hours_worked_today += work_done
            self.hours_worked_this_week += work_done
            task.status = TaskStatus.WRITING_IN_PROGRESS

            if task.writing_progress_hours >= self.actual_writing_hours:
                task.quality_score = self.actual_average_initial_quality
                task.update_from_quality()
                task.status = TaskStatus.COMPLETE
                self.current_task_id = None
                self.current_phase = None
            else:
                self.current_task_id = task.id
                self.current_phase = TaskStatus.WRITING_IN_PROGRESS
            return True

        elif task.status in (TaskStatus.NEEDS_WORK, TaskStatus.REVISION_IN_PROGRESS):
            # Revision work
            needed = self.actual_revision_hours - task.revision_progress_hours
            work_done = min(available_increment, needed)
            task.revision_progress_hours += work_done
            self.hours_worked_today += work_done
            self.hours_worked_this_week += work_done
            task.status = TaskStatus.REVISION_IN_PROGRESS

            if task.revision_progress_hours >= self.actual_revision_hours:
                task.quality_score = min(
                    1.0, task.quality_score + self.actual_revision_improvement
                )
                task.revision_count += 1
                task.update_from_quality()
                task.status = TaskStatus.FIXING_DONE
                # Clear reviewer assignment for new review cycle
                task.reviewer_id = None
                task.reviewer_domain = None
                task.review_progress_hours = 0.0
                self.current_task_id = None
                self.current_phase = None
            else:
                self.current_task_id = task.id
                self.current_phase = TaskStatus.REVISION_IN_PROGRESS
            return True

        return False

    def work_on_review_task(self, task: Task, available_increment: float) -> bool:
        """Performs review work on a task."""
        if task.status in (
            TaskStatus.COMPLETE,
            TaskStatus.FIXING_DONE,
            TaskStatus.REVIEW_IN_PROGRESS,
        ):
            # If just starting review, claim it and update review count
            if task.status != TaskStatus.REVIEW_IN_PROGRESS:
                task.status = TaskStatus.REVIEW_IN_PROGRESS
                task.reviewer_id = self.id
                task.reviewer_domain = self.domain_name

                # Increment review count for this trainer
                trainer_id = task.owner_id
                if trainer_id:
                    self.trainer_review_counts[trainer_id] = (
                        self.trainer_review_counts.get(trainer_id, 0) + 1
                    )

            # Calculate effective review time with decay
            trainer_id = task.owner_id
            effective_review_hours = (
                self.get_effective_review_time(trainer_id)
                if trainer_id
                else self.actual_review_hours
            )

            needed = effective_review_hours - task.review_progress_hours
            work_done = min(available_increment, needed)
            task.review_progress_hours += work_done
            self.hours_worked_today += work_done
            self.hours_worked_this_week += work_done

            if task.review_progress_hours >= effective_review_hours:
                if task.quality_score >= self.actual_quality_threshold:
                    task.status = TaskStatus.SIGNED_OFF
                else:
                    task.status = TaskStatus.NEEDS_WORK
                task.update_from_quality()
                self.current_task_id = None
                self.current_phase = None
            else:
                self.current_task_id = task.id
                self.current_phase = TaskStatus.REVIEW_IN_PROGRESS
            return True

        return False


@dataclass
class Simulation:
    """Manages and executes the task simulation process.

    Attributes:
        config: The SimulationConfig object containing all simulation parameters.
        agents: List of Agent instances.
        tasks: List of Task instances currently in the simulation.
        day: Current day of the simulation (1-indexed).
    """

    config: SimulationConfig
    agents: List[Agent] = field(default_factory=list, init=False)
    tasks: List[Task] = field(default_factory=list, init=False)
    day: int = field(default=0, init=False)
    _task_counter: int = field(
        default=0, init=False
    )  # Add counter for deterministic task IDs

    def __post_init__(self):
        """Initialize empty lists. Agents will be created in run() after seeding."""
        self.agents = []
        self.tasks = []
        self.day = 0
        self._task_counter = 0

    def _seed_random_generators(self):
        """Seeds both random and numpy.random generators if a seed is provided."""
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def _get_next_task_id(self) -> str:
        """Generates a deterministic task ID."""
        self._task_counter += 1
        return f"task_{self._task_counter:06d}"

    def _initialize_agents(self):
        """Initializes agent instances based on the simulation config."""
        self.agents = []
        trainer_counter = 1
        reviewer_counter = 1
        for domain_setup in self.config.domain_setups:
            for _ in range(domain_setup.num_top_performers):
                self.agents.append(
                    Agent(
                        id=f"Trainer{trainer_counter}_{domain_setup.domain_name[:3]}",
                        domain_name=domain_setup.domain_name,
                        performance_level="top_performer",
                        cfg=domain_setup.top_performer_cfg,
                    )
                )
                trainer_counter += 1
            for _ in range(domain_setup.num_normal_contractors):
                self.agents.append(
                    Agent(
                        id=f"NormalContractor{reviewer_counter}_{domain_setup.domain_name[:3]}",
                        domain_name=domain_setup.domain_name,
                        performance_level="normal_contractor",
                        cfg=domain_setup.normal_contractor_cfg,
                    )
                )
                reviewer_counter += 1
            for _ in range(domain_setup.num_bad_contractors):
                self.agents.append(
                    Agent(
                        id=f"BadContractor{reviewer_counter}_{domain_setup.domain_name[:3]}",
                        domain_name=domain_setup.domain_name,
                        performance_level="bad_contractor",
                        cfg=domain_setup.bad_contractor_cfg,
                    )
                )
                reviewer_counter += 1

    def reset_agents_weekly_hours(self):
        """Resets weekly worked hours for all agents."""
        for agent in self.agents:
            agent.reset_weekly_hours()

    def reset_agents_daily_hours(self):
        """Resets daily worked hours for all agents."""
        for agent in self.agents:
            agent.reset_daily_hours()

    def _process_agent_actions(
        self,
        agent: Agent,
        tasks_completed_writing_today: int,
        tasks_completed_revision_today: int,
        new_tasks_created_today: int,
        tasks_decisioned_today: int,
    ) -> tuple[bool, int, int, int, int]:
        """Processes actions for a single agent for one work increment opportunity."""
        action_taken_this_increment = False
        available_increment = agent.get_available_time_increment()
        if available_increment <= 0:
            return (
                False,
                tasks_completed_writing_today,
                tasks_completed_revision_today,
                new_tasks_created_today,
                tasks_decisioned_today,
            )

        # 1. Prioritize current task if any
        if agent.current_task_id and agent.current_phase:
            current_task_obj = next(
                (t for t in self.tasks if t.id == agent.current_task_id), None
            )
            if current_task_obj and current_task_obj.status == agent.current_phase:
                # Continue working on current task (writing or reviewing)
                if agent.current_phase in (
                    TaskStatus.WRITING_IN_PROGRESS,
                    TaskStatus.REVISION_IN_PROGRESS,
                ):
                    if agent.work_on_writing_task(
                        current_task_obj, available_increment
                    ):
                        action_taken_this_increment = True
                        if current_task_obj.status == TaskStatus.COMPLETE:
                            tasks_completed_writing_today += 1
                        if current_task_obj.status == TaskStatus.FIXING_DONE:
                            tasks_completed_revision_today += 1
                elif agent.current_phase == TaskStatus.REVIEW_IN_PROGRESS:
                    if agent.work_on_review_task(current_task_obj, available_increment):
                        action_taken_this_increment = True
                        if current_task_obj.status in (
                            TaskStatus.SIGNED_OFF,
                            TaskStatus.NEEDS_WORK,
                        ):
                            tasks_decisioned_today += 1
            else:  # Task no longer in the expected state
                agent.current_task_id = None
                agent.current_phase = None

        if (
            action_taken_this_increment
        ):  # If work was done on current task, agent's turn ends
            return (
                True,
                tasks_completed_writing_today,
                tasks_completed_revision_today,
                new_tasks_created_today,
                tasks_decisioned_today,
            )

        # 2. Decide whether to review or write for this increment
        should_review = agent.should_review_this_increment()

        if should_review:
            # Look for review work
            tasks_for_review = [
                t
                for t in self.tasks
                if (
                    (
                        t.status == TaskStatus.REVIEW_IN_PROGRESS
                        and t.reviewer_id == agent.id
                        and t.id != agent.current_task_id
                    )
                    or (
                        (
                            t.status == TaskStatus.COMPLETE
                            or t.status == TaskStatus.FIXING_DONE
                        )
                        and t.owner_domain == agent.domain_name
                        and t.reviewer_id
                        is None  # Only tasks not yet assigned to a reviewer
                    )
                )
            ]
            tasks_for_review.sort(
                key=lambda t: (
                    not (
                        t.status == TaskStatus.REVIEW_IN_PROGRESS
                        and t.reviewer_id == agent.id
                    ),
                    t.status != TaskStatus.COMPLETE,
                    t.id,
                )
            )

            if tasks_for_review:
                task_to_review = tasks_for_review[0]
                available_increment_for_review = agent.get_available_time_increment()
                if available_increment_for_review > 0 and agent.work_on_review_task(
                    task_to_review, available_increment_for_review
                ):
                    action_taken_this_increment = True
                    if task_to_review.status in (
                        TaskStatus.SIGNED_OFF,
                        TaskStatus.NEEDS_WORK,
                    ):
                        tasks_decisioned_today += 1

        if action_taken_this_increment:  # If review was done, agent's turn ends
            return (
                True,
                tasks_completed_writing_today,
                tasks_completed_revision_today,
                new_tasks_created_today,
                tasks_decisioned_today,
            )

        # 3. Look for writing work (revisions first, then new tasks)
        if np.random.random() < agent.cfg.revision_priority:
            tasks_needing_revision_by_agent = [
                t
                for t in self.tasks
                if t.owner_id == agent.id
                and (
                    t.status == TaskStatus.NEEDS_WORK
                    or (
                        t.status == TaskStatus.REVISION_IN_PROGRESS
                        and t.id != agent.current_task_id
                    )
                )
            ]
            tasks_needing_revision_by_agent.sort(
                key=lambda t: (t.status != TaskStatus.REVISION_IN_PROGRESS, t.id)
            )

            if tasks_needing_revision_by_agent:
                task_to_revise = tasks_needing_revision_by_agent[0]
                available_increment_for_revise = agent.get_available_time_increment()
                if available_increment_for_revise > 0 and agent.work_on_writing_task(
                    task_to_revise, available_increment_for_revise
                ):
                    action_taken_this_increment = True
                    if task_to_revise.status == TaskStatus.FIXING_DONE:
                        tasks_completed_revision_today += 1

        if action_taken_this_increment:  # If revision was done, agent's turn ends
            return (
                True,
                tasks_completed_writing_today,
                tasks_completed_revision_today,
                new_tasks_created_today,
                tasks_decisioned_today,
            )

        # 4. Look for new tasks to create/continue if no revision or priority not met
        tasks_to_write_by_agent = [
            t
            for t in self.tasks
            if t.owner_id == agent.id
            and (
                t.status == TaskStatus.CLAIMED
                or (
                    t.status == TaskStatus.WRITING_IN_PROGRESS
                    and t.id != agent.current_task_id
                )
            )
        ]
        tasks_to_write_by_agent.sort(
            key=lambda t: (t.status != TaskStatus.WRITING_IN_PROGRESS, t.id)
        )

        if tasks_to_write_by_agent:
            task_to_write = tasks_to_write_by_agent[0]
            available_increment_for_write = agent.get_available_time_increment()
            if available_increment_for_write > 0 and agent.work_on_writing_task(
                task_to_write, available_increment_for_write
            ):
                action_taken_this_increment = True
                if task_to_write.status == TaskStatus.COMPLETE:
                    tasks_completed_writing_today += 1
        else:  # No existing task to work on, create a new one
            new_task_object = agent.create_task(self._get_next_task_id())
            if new_task_object:
                self.tasks.append(new_task_object)
                new_tasks_created_today += 1
                # Try to work on it in the same increment
                available_increment_for_create = agent.get_available_time_increment()
                if available_increment_for_create > 0 and agent.work_on_writing_task(
                    new_task_object, available_increment_for_create
                ):
                    action_taken_this_increment = True
                    if new_task_object.status == TaskStatus.COMPLETE:
                        tasks_completed_writing_today += 1

        return (
            action_taken_this_increment,
            tasks_completed_writing_today,
            tasks_completed_revision_today,
            new_tasks_created_today,
            tasks_decisioned_today,
        )

    def _collect_daily_metrics(
        self,
        new_tasks_created_today: int,
        tasks_completed_writing_today: int,
        tasks_completed_revision_today: int,
        tasks_decisioned_today: int,
    ) -> dict:
        """Collects and returns a dictionary of daily summary metrics."""
        signed_off_cumulative = sum(
            1 for t in self.tasks if t.status == TaskStatus.SIGNED_OFF
        )
        tasks_claimed_eod = sum(1 for t in self.tasks if t.status == TaskStatus.CLAIMED)
        tasks_writing_in_progress_eod = sum(
            1 for t in self.tasks if t.status == TaskStatus.WRITING_IN_PROGRESS
        )
        tasks_needing_work_eod = sum(
            1 for t in self.tasks if t.status == TaskStatus.NEEDS_WORK
        )
        tasks_revision_in_progress_eod = sum(
            1 for t in self.tasks if t.status == TaskStatus.REVISION_IN_PROGRESS
        )
        tasks_review_in_progress_eod = sum(
            1 for t in self.tasks if t.status == TaskStatus.REVIEW_IN_PROGRESS
        )
        tasks_complete_waiting_review_eod = sum(
            1 for t in self.tasks if t.status == TaskStatus.COMPLETE
        )
        tasks_fixing_done_waiting_review_eod = sum(
            1 for t in self.tasks if t.status == TaskStatus.FIXING_DONE
        )

        avg_quality_signed_off = (
            np.mean(
                [
                    t.quality_score
                    for t in self.tasks
                    if t.status == TaskStatus.SIGNED_OFF
                ]
            )
            if any(t.status == TaskStatus.SIGNED_OFF for t in self.tasks)
            else 0.0
        )
        avg_agent_hrs_worked_today = (
            np.mean([a.hours_worked_today for a in self.agents]) if self.agents else 0.0
        )

        metrics = {
            "day": self.day,
            "new_tasks_created": new_tasks_created_today,
            "tasks_writing_completed": tasks_completed_writing_today,
            "tasks_revision_completed": tasks_completed_revision_today,
            "tasks_decisioned_by_review": tasks_decisioned_today,
            "signed_off_cumulative": signed_off_cumulative,
            "tasks_claimed_eod": tasks_claimed_eod,
            "tasks_writing_in_progress_eod": tasks_writing_in_progress_eod,
            "tasks_needing_work_eod": tasks_needing_work_eod,
            "tasks_revision_in_progress_eod": tasks_revision_in_progress_eod,
            "tasks_review_in_progress_eod": tasks_review_in_progress_eod,
            "tasks_complete_waiting_review_eod": tasks_complete_waiting_review_eod,
            "tasks_fixing_done_waiting_review_eod": tasks_fixing_done_waiting_review_eod,
            "avg_quality_signed_off": avg_quality_signed_off,
            "avg_agent_hrs_worked_today": avg_agent_hrs_worked_today,
            "total_tasks_in_system": len(self.tasks),
        }
        # Add the seed used for this simulation run to the metrics for the first day only
        if self.day == 1:
            metrics["simulation_random_seed"] = self.config.random_seed
        return metrics

    def run(self):
        """Runs the simulation for the configured number of days.

        Handles daily agent actions (task creation, revision, review) based on their
        availability and priorities. Collects daily summary statistics.

        Returns:
            A pandas DataFrame containing the daily summary of the simulation.
        """
        self._seed_random_generators()
        self._initialize_agents()

        daily_summary_data = []
        # work_increment = WORK_INCREMENT_HOURS # Constant, can be used directly

        for self.day in range(1, self.config.simulation_days + 1):
            if (self.day - 1) % self.config.week_length_days == 0:
                self.reset_agents_weekly_hours()

            self.reset_agents_daily_hours()

            # Daily counters for metrics
            new_tasks_created_today = 0
            tasks_completed_writing_today = 0
            tasks_completed_revision_today = 0
            tasks_decisioned_today = 0

            # Shuffle agents using numpy permutation for determinism
            agent_indices = np.random.permutation(len(self.agents))
            self.agents = [self.agents[i] for i in agent_indices]

            agents_still_working_in_day = True
            while agents_still_working_in_day:
                agents_still_working_in_day = (
                    False  # Assume no one works this sub-increment pass
                )

                # Process Agents
                for agent in self.agents:
                    (
                        action_taken,
                        tasks_completed_writing_today,
                        tasks_completed_revision_today,
                        new_tasks_created_today,
                        tasks_decisioned_today,
                    ) = self._process_agent_actions(
                        agent,
                        tasks_completed_writing_today,
                        tasks_completed_revision_today,
                        new_tasks_created_today,
                        tasks_decisioned_today,
                    )
                    if action_taken:
                        agents_still_working_in_day = True

            daily_summary_data.append(
                self._collect_daily_metrics(
                    new_tasks_created_today,
                    tasks_completed_writing_today,
                    tasks_completed_revision_today,
                    tasks_decisioned_today,
                )
            )

        # Fill forward the seed for all days in the summary for easy access
        df = pd.DataFrame(daily_summary_data)
        if "simulation_random_seed" in df.columns:
            df["simulation_random_seed"] = df["simulation_random_seed"].ffill()
        return df
