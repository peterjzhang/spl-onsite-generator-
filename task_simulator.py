from dataclasses import dataclass, field
from enum import Enum, auto
import uuid
import numpy as np
import pandas as pd
import random
from typing import List, Optional, Union
# import ace_tools as tools # Commented out as it's used for demo, not core sim logic

WORK_INCREMENT_HOURS = 0.5 # Minimum work block

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
class TrainerConfig:
    """Configuration settings for Trainer agents.

    Attributes:
        max_hours_per_week: Maximum hours a trainer can work in a week.
        target_hours_per_day: Target daily work hours for a trainer.
        target_hours_per_day_noise: Standard deviation for daily target hours noise.
        writing_hours: Base hours required to write a new task.
        writing_hours_noise: Standard deviation for task writing hours noise.
        revision_hours: Base hours required to revise a task.
        revision_hours_noise: Standard deviation for task revision hours noise.
        average_initial_quality: Average quality score of initially created tasks (0.0 to 1.0).
        average_initial_quality_noise: Standard deviation for initial quality noise.
        revision_improvement: Average improvement in quality score after a revision.
        revision_improvement_noise: Standard deviation for revision improvement noise.
        revision_priority: Probability (0.0 to 1.0) a trainer will prioritize revising
                           an existing task over creating a new one.
    """
    max_hours_per_week: float = 20.0
    target_hours_per_day: float = 4.0
    target_hours_per_day_noise: float = 0.0
    writing_hours: float = 6.0
    writing_hours_noise: float = 0.0
    revision_hours: float = 1.5
    revision_hours_noise: float = 0.0
    average_initial_quality: float = 0.7
    average_initial_quality_noise: float = 0.0
    revision_improvement: float = 0.1
    revision_improvement_noise: float = 0.0
    revision_priority: float = 0.7
    # domain_name: Optional[str] = None # Domain name will be on agent, config is generic template


@dataclass
class ReviewerConfig:
    """Configuration settings for Reviewer agents.

    Attributes:
        max_hours_per_week: Maximum hours a reviewer can work in a week.
        target_hours_per_day: Target daily work hours for a reviewer.
        target_hours_per_day_noise: Standard deviation for daily target hours noise.
        review_hours: Base hours required to review a task.
        review_hours_noise: Standard deviation for task review hours noise.
        quality_threshold: Minimum quality score for a task to be signed off.
        quality_threshold_noise: Standard deviation for quality threshold noise.
    """
    max_hours_per_week: float = 40.0
    target_hours_per_day: float = 8.0
    target_hours_per_day_noise: float = 0.0
    review_hours: float = 2.0
    review_hours_noise: float = 0.0
    quality_threshold: float = 0.8
    quality_threshold_noise: float = 0.0
    # domain_name: Optional[str] = None # Domain name will be on agent, config is generic template


@dataclass
class DomainSimulationSetup:
    """Configuration for a specific domain within a multi-domain simulation.

    Attributes:
        domain_name: The name of the domain.
        num_trainers: Number of trainer agents in this domain.
        num_reviewers: Number of reviewer agents in this domain.
        trainer_cfg: Configuration object for trainers in this domain.
        reviewer_cfg: Configuration object for reviewers in this domain.
    """
    domain_name: str
    num_trainers: int
    num_reviewers: int
    trainer_cfg: TrainerConfig
    reviewer_cfg: ReviewerConfig
    # hourly_rate: Optional[float] = None # For future cost analysis


@dataclass
class SimulationConfig:
    """Overall configuration for the simulation.

    Attributes:
        simulation_days: Total number of days the simulation will run.
        week_length_days: Number of days in a work week (for resetting weekly hours).
        domain_setups: A list of DomainSimulationSetup objects, defining each
                       domain to be simulated.
    """
    # num_trainers: int = 3 # Replaced by domain_configs
    # num_reviewers: int = 2 # Replaced by domain_configs
    simulation_days: int = 21
    week_length_days: int = 7
    # trainer_cfg: TrainerConfig = field(default_factory=TrainerConfig) # Replaced
    # reviewer_cfg: ReviewerConfig = field(default_factory=ReviewerConfig) # Replaced
    domain_setups: List[DomainSimulationSetup] = field(default_factory=list)


@dataclass
class Task:
    """Represents a single task within the simulation.

    Attributes:
        id: Unique identifier for the task.
        owner_id: ID of the TrainerAgent who created the task.
        owner_domain: Domain of the trainer who created the task.
        reviewer_id: ID of the ReviewerAgent who reviewed/is reviewing the task.
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
    owner_domain: Optional[str] = None # Added to track task origin domain
    reviewer_id: Optional[str] = None
    reviewer_domain: Optional[str] = None # Added to track reviewer domain if different
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
class BaseAgent:
    """Base class for all agents in the simulation.

    Attributes:
        id: Unique identifier for the agent.
        domain_name: The domain this agent belongs to.
        cfg: Configuration object (TrainerConfig or ReviewerConfig).
        actual_target_hours_per_day: The agent's specific target hours for the day, 
                                     sampled based on config noise.
        hours_worked_this_week: Total hours worked by the agent in the current week.
        hours_worked_today: Total hours worked by the agent today.
        current_task_id: ID of the task the agent is currently focused on.
        current_phase: The specific phase of work the agent is performing on the current_task_id.
    """
    id: str
    domain_name: str # Added domain for each agent
    cfg: Union[TrainerConfig, ReviewerConfig] # Moved cfg to BaseAgent
    actual_target_hours_per_day: float = field(init=False) # Moved from children
    hours_worked_this_week: float = 0.0
    hours_worked_today: float = 0.0
    # Store current task and phase to prioritize continuation
    current_task_id: Optional[str] = None
    current_phase: Optional[TaskStatus] = None # e.g. WRITING_IN_PROGRESS, REVISION_IN_PROGRESS, REVIEW_IN_PROGRESS

    def __post_init__(self):
        """Initializes actual agent parameters based on config noise."""
        self.actual_target_hours_per_day = max(0.1, np.random.normal(self.cfg.target_hours_per_day, self.cfg.target_hours_per_day_noise))

    def reset_weekly_hours(self):
        """Resets the agent's weekly worked hours count."""
        self.hours_worked_this_week = 0.0

    def reset_daily_hours(self):
        """Resets the agent's daily worked hours count."""
        self.hours_worked_today = 0.0
        # self.current_task_id = None # Optionally reset current task at day end, or let it persist
        # self.current_phase = None

    def get_available_time_increment(self) -> float:
        """Calculates the available time for the agent to work in the next increment.
        
        Returns the minimum of WORK_INCREMENT_HOURS, remaining daily hours, 
        and remaining weekly hours.
        """
        remaining_today = self.actual_target_hours_per_day - self.hours_worked_today
        remaining_week = self.cfg.max_hours_per_week - self.hours_worked_this_week
        
        if remaining_today <= 0 or remaining_week <= 0:
            return 0.0
            
        return min(WORK_INCREMENT_HOURS, remaining_today, remaining_week)


@dataclass
class TrainerAgent(BaseAgent):
    """Represents a Trainer agent who creates and revises tasks.

    Inherits from BaseAgent.

    Attributes:
        actual_writing_hours: Actual hours this trainer takes to write a task (sampled).
        actual_revision_hours: Actual hours this trainer takes to revise a task (sampled).
        actual_average_initial_quality: Actual initial quality of tasks by this trainer (sampled).
        actual_revision_improvement: Actual quality improvement per revision by this trainer (sampled).
    """
    # cfg: TrainerConfig = field(kw_only=True) # Moved to BaseAgent, will be specialized type there
    # actual_target_hours_per_day: float = field(init=False) # Moved to BaseAgent
    actual_writing_hours: float = field(init=False)
    actual_revision_hours: float = field(init=False)
    actual_average_initial_quality: float = field(init=False)
    actual_revision_improvement: float = field(init=False)

    def __post_init__(self):
        """Initializes actual trainer-specific parameters based on config noise."""
        super().__post_init__() # Call BaseAgent's post_init
        # Type assertion for self.cfg for type checker, assuming cfg is always TrainerConfig for TrainerAgent
        trainer_cfg: TrainerConfig = self.cfg # type: ignore 
        self.actual_writing_hours = max(0.1, np.random.normal(trainer_cfg.writing_hours, trainer_cfg.writing_hours_noise))
        self.actual_revision_hours = max(0.1, np.random.normal(trainer_cfg.revision_hours, trainer_cfg.revision_hours_noise))
        self.actual_average_initial_quality = np.clip(np.random.normal(trainer_cfg.average_initial_quality, trainer_cfg.average_initial_quality_noise), 0.0, 1.0)
        self.actual_revision_improvement = max(0.0, np.random.normal(trainer_cfg.revision_improvement, trainer_cfg.revision_improvement_noise))

    def can_work(self, hours_for_task: float) -> bool:
        """Checks if the trainer has any available time to work.
        
        Note: This is less central now as work is done in increments controlled by
        get_available_time_increment.
        """
        # This method is now less central as work is done in increments.
        # get_available_time_increment will determine actual workable time.
        # However, it can still be used for initial check if a task *could* be started.
        remaining_today = self.actual_target_hours_per_day - self.hours_worked_today
        # Type assertion for self.cfg for type checker
        trainer_cfg: TrainerConfig = self.cfg # type: ignore
        remaining_week = trainer_cfg.max_hours_per_week - self.hours_worked_this_week
        
        if remaining_today <= 0 or remaining_week <= 0:
            return False
        return min(remaining_today, remaining_week) > 0

    def work_on_task(self, task: Task, available_increment: float) -> bool:
        """Performs a work increment on a given task (either writing or revising).

        Updates task progress, status, quality, and agent's worked hours.

        Args:
            task: The Task object to work on.
            available_increment: The amount of time the agent can work in this increment.

        Returns:
            True if work was done, False otherwise.
        """
        if task.status in (TaskStatus.CLAIMED, TaskStatus.WRITING_IN_PROGRESS):
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
                self.current_task_id = None # Phase complete
                self.current_phase = None
            else: # Still in progress
                self.current_task_id = task.id
                self.current_phase = TaskStatus.WRITING_IN_PROGRESS
            return True

        elif task.status in (TaskStatus.NEEDS_WORK, TaskStatus.REVISION_IN_PROGRESS):
            needed = self.actual_revision_hours - task.revision_progress_hours
            work_done = min(available_increment, needed)
            task.revision_progress_hours += work_done
            self.hours_worked_today += work_done
            self.hours_worked_this_week += work_done
            task.status = TaskStatus.REVISION_IN_PROGRESS
            if task.revision_progress_hours >= self.actual_revision_hours:
                task.quality_score = min(1.0, task.quality_score + self.actual_revision_improvement)
                task.revision_count += 1
                task.update_from_quality()
                task.status = TaskStatus.FIXING_DONE
                self.current_task_id = None # Phase complete
                self.current_phase = None
            else: # Still in progress
                self.current_task_id = task.id
                self.current_phase = TaskStatus.REVISION_IN_PROGRESS
            return True
        return False

    def create_task(self) -> Optional[Task]:
        """Creates a new Task object, assigning self as owner and current domain.
        
        The task is initially in CLAIMED status. Actual work (writing) starts
        via work_on_task.
        
        Returns:
            A new Task object, or None if creation fails (though current logic always succeeds).
        """
        # This method will now mostly just claim a task. The work is done in work_on_task.
        # A new task is created if the agent decides to start one.
        # It doesn't consume all writing hours at once.
        task = Task(
            id=str(uuid.uuid4()),
            owner_id=self.id,
            owner_domain=self.domain_name, # Assign trainer's domain to task
            status=TaskStatus.CLAIMED # Initial status
        )
        # Initial quality is set when writing is complete.
        # Hours are not added here, they are added in work_on_task
        return task

    def revise_task(self, task: Task) -> bool:
        """Sets a task's status to REVISION_IN_PROGRESS if it NEEDS_WORK.

        Note: This method is less central now. The main revision logic is handled 
        within work_on_task when a task is in NEEDS_WORK or REVISION_IN_PROGRESS status.
        It can be used to explicitly mark a task for revision if needed by external logic.

        Args:
            task: The task to be revised.

        Returns:
            True if the task's status was set to REVISION_IN_PROGRESS or was already in that state, 
            False otherwise.
        """
        # This method is now less relevant, work_on_task handles revision.
        # Kept for potential direct calls or logic, but main flow is via work_on_task.
        if task.status == TaskStatus.NEEDS_WORK:
            task.status = TaskStatus.REVISION_IN_PROGRESS # Mark as starting revision
            self.current_task_id = task.id
            self.current_phase = TaskStatus.REVISION_IN_PROGRESS
            return True # Indicates revision process has started
        elif task.status == TaskStatus.REVISION_IN_PROGRESS:
            return True # Already in progress
        return False


@dataclass
class ReviewerAgent(BaseAgent):
    """Represents a Reviewer agent who reviews tasks and makes decisions.

    Inherits from BaseAgent.

    Attributes:
        actual_review_hours: Actual hours this reviewer takes to review a task (sampled).
        actual_quality_threshold: Actual quality threshold this reviewer uses for sign-off (sampled).
    """
    # cfg: ReviewerConfig = field(kw_only=True) # Moved to BaseAgent
    # actual_target_hours_per_day: float = field(init=False) # Moved to BaseAgent
    actual_review_hours: float = field(init=False)
    actual_quality_threshold: float = field(init=False)

    def __post_init__(self):
        """Initializes actual reviewer-specific parameters based on config noise."""
        super().__post_init__() # Call BaseAgent's post_init
        # Type assertion for self.cfg for type checker
        reviewer_cfg: ReviewerConfig = self.cfg # type: ignore
        self.actual_review_hours = max(0.1, np.random.normal(reviewer_cfg.review_hours, reviewer_cfg.review_hours_noise))
        self.actual_quality_threshold = np.clip(np.random.normal(reviewer_cfg.quality_threshold, reviewer_cfg.quality_threshold_noise), 0.0, 1.0)

    def can_work(self, hours_for_task: float) -> bool:
        """Checks if the reviewer has any available time to work.
        
        Note: This is less central now as work is done in increments controlled by
        get_available_time_increment.
        """
        # Similar to TrainerAgent, this is less central now.
        remaining_today = self.actual_target_hours_per_day - self.hours_worked_today
        # Type assertion for self.cfg for type checker
        reviewer_cfg: ReviewerConfig = self.cfg # type: ignore
        remaining_week = reviewer_cfg.max_hours_per_week - self.hours_worked_this_week
        if remaining_today <= 0 or remaining_week <= 0:
            return False
        return min(remaining_today, remaining_week) > 0

    def work_on_review(self, task: Task, available_increment: float) -> bool:
        """Performs a work increment on reviewing a given task.

        Updates task progress, status (to SIGNED_OFF or NEEDS_WORK if review completes),
        assigns reviewer_id and reviewer_domain if starting review. Updates agent's worked hours.

        Args:
            task: The Task object to review.
            available_increment: The amount of time the agent can work in this increment.

        Returns:
            True if work was done, False otherwise.
        """
        if task.status in (TaskStatus.COMPLETE, TaskStatus.FIXING_DONE, TaskStatus.REVIEW_IN_PROGRESS):
            if task.status != TaskStatus.REVIEW_IN_PROGRESS: # If just starting review on this task
                 task.status = TaskStatus.REVIEW_IN_PROGRESS
                 task.reviewer_id = self.id # Claim the review
                 task.reviewer_domain = self.domain_name # Assign reviewer's domain

            needed = self.actual_review_hours - task.review_progress_hours
            work_done = min(available_increment, needed)
            task.review_progress_hours += work_done
            self.hours_worked_today += work_done
            self.hours_worked_this_week += work_done
            
            if task.review_progress_hours >= self.actual_review_hours:
                if task.quality_score >= self.actual_quality_threshold:
                    task.status = TaskStatus.SIGNED_OFF
                else:
                    task.status = TaskStatus.NEEDS_WORK
                task.update_from_quality() # Update issues based on final quality
                self.current_task_id = None # Phase complete
                self.current_phase = None
            else: # Still in progress
                self.current_task_id = task.id
                self.current_phase = TaskStatus.REVIEW_IN_PROGRESS
            return True
        return False

    def review_task(self, task: Task) -> bool:
        """Sets a task's status to REVIEW_IN_PROGRESS if it's ready for review.
        
        Assigns self as reviewer and sets reviewer_domain. 
        Note: This method is less central now. The main review logic is handled 
        within work_on_review. It can be used to explicitly mark a task for review 
        if needed by external logic.

        Args:
            task: The task to be reviewed.

        Returns:
            True if the task's status was set to REVIEW_IN_PROGRESS or was already in that state, 
            False otherwise.
        """
        # This method is now less relevant, work_on_review handles it.
        # Kept for potential direct calls, but main flow is via work_on_review.
        if task.status in (TaskStatus.COMPLETE, TaskStatus.FIXING_DONE):
            task.status = TaskStatus.REVIEW_IN_PROGRESS # Mark as starting review
            task.reviewer_id = self.id
            task.reviewer_domain = self.domain_name # Assign reviewer's domain
            self.current_task_id = task.id
            self.current_phase = TaskStatus.REVIEW_IN_PROGRESS
            return True # Indicates review process has started
        elif task.status == TaskStatus.REVIEW_IN_PROGRESS:
            return True # Already in progress
        return False
        

@dataclass
class Simulation:
    """Manages and executes the task simulation process.

    Attributes:
        config: The SimulationConfig object containing all simulation parameters.
        trainers: List of TrainerAgent instances.
        reviewers: List of ReviewerAgent instances.
        tasks: List of Task instances currently in the simulation.
        day: Current day of the simulation (1-indexed).
    """
    config: SimulationConfig
    trainers: List[TrainerAgent] = field(default_factory=list, init=False)
    reviewers: List[ReviewerAgent] = field(default_factory=list, init=False)
    tasks: List[Task] = field(default_factory=list, init=False)
    day: int = field(default=0, init=False)

    def __post_init__(self):
        """Initializes trainer and reviewer agents based on the simulation config."""
        self.trainers = []
        self.reviewers = []
        trainer_counter = 1
        reviewer_counter = 1
        for domain_setup in self.config.domain_setups:
            for _ in range(domain_setup.num_trainers):
                self.trainers.append(
                    TrainerAgent(
                        id=f"Trainer{trainer_counter}_{domain_setup.domain_name[:3]}", 
                        domain_name=domain_setup.domain_name,
                        cfg=domain_setup.trainer_cfg
                    )
                )
                trainer_counter +=1
            for _ in range(domain_setup.num_reviewers):
                self.reviewers.append(
                    ReviewerAgent(
                        id=f"Reviewer{reviewer_counter}_{domain_setup.domain_name[:3]}", 
                        domain_name=domain_setup.domain_name,
                        cfg=domain_setup.reviewer_cfg
                    )
                )
                reviewer_counter +=1

    def reset_agents_weekly_hours(self):
        """Resets weekly worked hours for all agents."""
        for agent in self.trainers + self.reviewers:
            agent.reset_weekly_hours()

    def reset_agents_daily_hours(self):
        """Resets daily worked hours for all agents."""
        for agent in self.trainers + self.reviewers:
            agent.reset_daily_hours()

    def _process_trainer_actions(self, trainer: TrainerAgent, tasks_completed_writing_today: int, tasks_completed_revision_today: int, new_tasks_created_today: int) -> tuple[bool, int, int, int]:
        """Processes actions for a single trainer for one work increment opportunity."""
        action_taken_this_increment = False
        available_increment = trainer.get_available_time_increment()
        if available_increment <= 0:
            return False, tasks_completed_writing_today, tasks_completed_revision_today, new_tasks_created_today

        # 1. Prioritize current task if any
        if trainer.current_task_id and trainer.current_phase:
            current_task_obj = next((t for t in self.tasks if t.id == trainer.current_task_id), None)
            if current_task_obj and current_task_obj.status == trainer.current_phase:
                if trainer.work_on_task(current_task_obj, available_increment):
                    action_taken_this_increment = True
                    if current_task_obj.status == TaskStatus.COMPLETE: tasks_completed_writing_today += 1
                    if current_task_obj.status == TaskStatus.FIXING_DONE: tasks_completed_revision_today += 1
            else: # Task no longer in the expected state
                trainer.current_task_id = None
                trainer.current_phase = None
        
        if action_taken_this_increment: # If work was done on current task, trainer's turn for this increment ends
            return True, tasks_completed_writing_today, tasks_completed_revision_today, new_tasks_created_today

        # 2. Look for revisions if no current task or current task changed state
        trainer_config_for_priority: TrainerConfig = trainer.cfg # type: ignore
        if random.random() < trainer_config_for_priority.revision_priority:
            tasks_needing_revision_by_trainer = [
                t for t in self.tasks 
                if t.owner_id == trainer.id and (t.status == TaskStatus.NEEDS_WORK or 
                                             (t.status == TaskStatus.REVISION_IN_PROGRESS and t.id != trainer.current_task_id))
            ]
            tasks_needing_revision_by_trainer.sort(key=lambda t: (t.status != TaskStatus.REVISION_IN_PROGRESS, t.id))

            if tasks_needing_revision_by_trainer:
                task_to_revise = tasks_needing_revision_by_trainer[0]
                # Use a fresh time increment check for this new action
                available_increment_for_revise = trainer.get_available_time_increment()
                if available_increment_for_revise > 0 and trainer.work_on_task(task_to_revise, available_increment_for_revise):
                    action_taken_this_increment = True
                    if task_to_revise.status == TaskStatus.FIXING_DONE: tasks_completed_revision_today += 1
        
        if action_taken_this_increment: # If revision was done, trainer's turn for this increment ends
            return True, tasks_completed_writing_today, tasks_completed_revision_today, new_tasks_created_today

        # 3. Look for new tasks to create/continue if no revision or priority not met
        tasks_to_write_by_trainer = [
            t for t in self.tasks
            if t.owner_id == trainer.id and (t.status == TaskStatus.CLAIMED or 
                                         (t.status == TaskStatus.WRITING_IN_PROGRESS and t.id != trainer.current_task_id))
        ]
        tasks_to_write_by_trainer.sort(key=lambda t: (t.status != TaskStatus.WRITING_IN_PROGRESS, t.id))

        if tasks_to_write_by_trainer:
            task_to_write = tasks_to_write_by_trainer[0]
            # Use a fresh time increment check for this new action
            available_increment_for_write = trainer.get_available_time_increment()
            if available_increment_for_write > 0 and trainer.work_on_task(task_to_write, available_increment_for_write):
                action_taken_this_increment = True
                if task_to_write.status == TaskStatus.COMPLETE: tasks_completed_writing_today += 1
        else: # No existing task to work on, create a new one
            new_task_object = trainer.create_task()
            if new_task_object:
                self.tasks.append(new_task_object)
                new_tasks_created_today += 1
                # Try to work on it in the same increment
                available_increment_for_create = trainer.get_available_time_increment()
                if available_increment_for_create > 0 and trainer.work_on_task(new_task_object, available_increment_for_create): 
                    action_taken_this_increment = True
                    if new_task_object.status == TaskStatus.COMPLETE: tasks_completed_writing_today += 1
        
        return action_taken_this_increment, tasks_completed_writing_today, tasks_completed_revision_today, new_tasks_created_today

    def _process_reviewer_actions(self, reviewer: ReviewerAgent, tasks_decisioned_today: int) -> tuple[bool, int]:
        """Processes actions for a single reviewer for one work increment opportunity."""
        action_taken_this_increment = False
        available_increment = reviewer.get_available_time_increment()
        if available_increment <= 0:
            return False, tasks_decisioned_today

        # 1. Prioritize current review task
        if reviewer.current_task_id and reviewer.current_phase:
            current_task_obj = next((t for t in self.tasks if t.id == reviewer.current_task_id), None)
            if current_task_obj and current_task_obj.status == reviewer.current_phase:
                if reviewer.work_on_review(current_task_obj, available_increment):
                    action_taken_this_increment = True
                    if current_task_obj.status in (TaskStatus.SIGNED_OFF, TaskStatus.NEEDS_WORK):
                        tasks_decisioned_today += 1
            else: # Task no longer in expected state
                reviewer.current_task_id = None
                reviewer.current_phase = None
        
        if action_taken_this_increment: # If work was done on current task, reviewer's turn for this increment ends
            return True, tasks_decisioned_today

        # 2. Find a new task to review if no current task
        tasks_for_reviewer = [
            t for t in self.tasks 
            if (
                (t.status == TaskStatus.REVIEW_IN_PROGRESS and t.reviewer_id == reviewer.id and t.id != reviewer.current_task_id) or
                ((t.status == TaskStatus.COMPLETE or t.status == TaskStatus.FIXING_DONE) and t.owner_domain == reviewer.domain_name)
               )
        ]
        tasks_for_reviewer.sort(key=lambda t: (
            not (t.status == TaskStatus.REVIEW_IN_PROGRESS and t.reviewer_id == reviewer.id),
            t.status != TaskStatus.COMPLETE,
            t.id
        ))

        if tasks_for_reviewer:
            task_to_review = tasks_for_reviewer[0]
            # Use a fresh time increment check for this new action
            available_increment_for_review = reviewer.get_available_time_increment()
            if available_increment_for_review > 0 and reviewer.work_on_review(task_to_review, available_increment_for_review):
                action_taken_this_increment = True
                if task_to_review.status in (TaskStatus.SIGNED_OFF, TaskStatus.NEEDS_WORK):
                     tasks_decisioned_today += 1
        
        return action_taken_this_increment, tasks_decisioned_today

    def _collect_daily_metrics(self, new_tasks_created_today: int, tasks_completed_writing_today: int, tasks_completed_revision_today: int, tasks_decisioned_today: int) -> dict:
        """Collects and returns a dictionary of daily summary metrics."""
        signed_off_cumulative = sum(1 for t in self.tasks if t.status == TaskStatus.SIGNED_OFF)
        tasks_claimed_eod = sum(1 for t in self.tasks if t.status == TaskStatus.CLAIMED)
        tasks_writing_in_progress_eod = sum(1 for t in self.tasks if t.status == TaskStatus.WRITING_IN_PROGRESS)
        tasks_needing_work_eod = sum(1 for t in self.tasks if t.status == TaskStatus.NEEDS_WORK)
        tasks_revision_in_progress_eod = sum(1 for t in self.tasks if t.status == TaskStatus.REVISION_IN_PROGRESS)
        tasks_review_in_progress_eod = sum(1 for t in self.tasks if t.status == TaskStatus.REVIEW_IN_PROGRESS)
        tasks_complete_waiting_review_eod = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETE)
        tasks_fixing_done_waiting_review_eod = sum(1 for t in self.tasks if t.status == TaskStatus.FIXING_DONE)

        avg_quality_signed_off = np.mean([t.quality_score for t in self.tasks if t.status == TaskStatus.SIGNED_OFF]) if any(t.status == TaskStatus.SIGNED_OFF for t in self.tasks) else 0.0
        avg_trainer_hrs_worked_today = np.mean([tr.hours_worked_today for tr in self.trainers]) if self.trainers else 0.0
        avg_reviewer_hrs_worked_today = np.mean([rev.hours_worked_today for rev in self.reviewers]) if self.reviewers else 0.0
        # Note: weekly averages were removed from daily summary based on prior user feedback for app.py plots

        return {
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
            "avg_trainer_hrs_worked_today": avg_trainer_hrs_worked_today,
            "avg_reviewer_hrs_worked_today": avg_reviewer_hrs_worked_today,
            "total_tasks_in_system": len(self.tasks)
        }

    def run(self):
        """Runs the simulation for the configured number of days.

        Handles daily agent actions (task creation, revision, review) based on their
        availability and priorities. Collects daily summary statistics.

        Returns:
            A pandas DataFrame containing the daily summary of the simulation.
        """
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

            random.shuffle(self.trainers)
            random.shuffle(self.reviewers)

            agents_still_working_in_day = True
            while agents_still_working_in_day:
                agents_still_working_in_day = False # Assume no one works this sub-increment pass

                # Process Trainers
                for trainer in self.trainers:
                    action_taken, tasks_completed_writing_today, tasks_completed_revision_today, new_tasks_created_today = \
                        self._process_trainer_actions(trainer, tasks_completed_writing_today, tasks_completed_revision_today, new_tasks_created_today)
                    if action_taken:
                        agents_still_working_in_day = True

                # Process Reviewers
                for reviewer in self.reviewers:
                    action_taken, tasks_decisioned_today = \
                        self._process_reviewer_actions(reviewer, tasks_decisioned_today)
                    if action_taken:
                        agents_still_working_in_day = True
            
            daily_summary_data.append(self._collect_daily_metrics(
                new_tasks_created_today, tasks_completed_writing_today, 
                tasks_completed_revision_today, tasks_decisioned_today
            ))

        return pd.DataFrame(daily_summary_data)