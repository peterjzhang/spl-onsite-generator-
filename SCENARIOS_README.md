# Pre-configured Simulation Scenarios

This document provides an intuitive explanation of the pre-configured simulation scenarios found in the `configs/` directory. Each scenario is designed to highlight different dynamics and bottlenecks within the task creation and review workflow.

To use these scenarios, you can load their respective `.json` files (e.g., [`lax_reviewers.json`](./configs/lax_reviewers.json)) via the "Load Preset for Single Config" option or assign them to domains in the "Multi-Domain Configuration" mode in the Streamlit application (`app.py`).

---

## 1. Lax Reviewers ([`lax_reviewers.json`](./configs/lax_reviewers.json))

-   **Core Idea**: This scenario models a situation where reviewers have very low standards for signing off tasks.
-   **Key Parameter Settings**:
    -   **ReviewerConfig (`reviewer_config`)**:
        -   `quality_threshold`: Set to a very low value (e.g., 0.1 or 0.2). This means tasks with poor quality can still be approved.
        -   `quality_threshold_noise`: Could be low to ensure most reviewers are consistently lax, or higher if some are lax and others aren't, creating a mix.
-   **Expected Outcomes**:
    -   **High Sign-off Rate**: Many tasks will likely be signed off quickly, regardless of their initial quality.
    -   **Low Average Quality of Signed-off Tasks**: The overall quality of completed work will be poor.
    -   **Few Revision Cycles**: Tasks will rarely be sent back for `NEEDS_WORK`, leading to minimal revisions.
    -   **Potential for High Throughput (but low quality)**: Reviewers might not be a bottleneck in terms of task flow, but the output quality suffers significantly.

---

## 2. Reviewer Bottleneck ([`reviewer_bottleneck.json`](./configs/reviewer_bottleneck.json))

-   **Core Idea**: Simulates a situation where there are not enough reviewers (or reviewers are too slow) to handle the volume of tasks produced by trainers.
-   **Key Parameter Settings**:
    -   **Simulation Settings (`simulation_settings`)**:
        -   `num_reviewers`: Set to a significantly lower number compared to `num_trainers`.
    -   **ReviewerConfig (`reviewer_config`)**:
        -   `review_hours`: Could be set relatively high, making each review take longer.
        -   `target_hours_per_day`: Could be low for reviewers, limiting their daily capacity.
-   **Expected Outcomes**:
    -   **Growing Queue of Tasks Awaiting Review**: The number of tasks in `COMPLETE` or `FIXING_DONE` status will steadily increase.
    -   **Low Reviewer Utilization (paradoxically at first, then high)**: Initially, reviewers might not have tasks, but once the queue forms, they will be 100% utilized but still unable to clear the backlog.
    -   **Delayed Sign-offs**: Tasks will take a long time to get through the review stage.
    -   Trainers might become idle if the backlog is severe and they have nothing to revise.

---

## 3. Writer Bottleneck ([`writer_bottleneck.json`](./configs/writer_bottleneck.json))

-   **Core Idea**: Simulates a situation where there are not enough trainers to produce tasks or handle revisions efficiently, or tasks take a very long time to write.
-   **Key Parameter Settings**:
    -   **Simulation Settings (`simulation_settings`)**:
        -   `num_trainers`: Set to a significantly lower number compared to `num_reviewers`.
    -   **TrainerConfig (`trainer_config`)**:
        -   `writing_hours`: Could be set very high.
        -   `revision_hours`: Could also be high, making revisions lengthy.
        -   `target_hours_per_day`: Could be low for trainers.
-   **Expected Outcomes**:
    -   **Low Number of New Tasks Created**: The system will be starved of new work.
    -   **Reviewers May Be Idle**: Reviewers might frequently have no tasks to review.
    -   **Slow Revision Turnaround**: If tasks do get sent for `NEEDS_WORK`, they might take a long time to be revised.
    -   Overall low system throughput.

---

## 4. Challenging Task ([`challenging_task.json`](./configs/challenging_task.json))

-   **Core Idea**: Models tasks that are inherently difficult, taking a long time to create and having low initial quality.
-   **Key Parameter Settings**:
    -   **TrainerConfig (`trainer_config`)**:
        -   `writing_hours`: Set to a very high value.
        -   `average_initial_quality`: Set to a very low value (e.g., 0.1-0.3).
        -   `revision_improvement`: Could be moderate or low, meaning even revisions don't improve quality drastically or quickly.
        -   `revision_hours`: Could be high if fixing these challenging tasks is also time-consuming.
-   **Expected Outcomes**:
    -   **Slow Task Creation**: Few new tasks enter the system due to long writing times.
    -   **Very Low Initial Quality**: Most tasks will be of poor quality when first submitted for review.
    -   **High `NEEDS_WORK` Rate**: Reviewers will likely send many tasks back for revision.
    -   **Multiple Revision Cycles**: Tasks may go through several rounds of revision before being signed off, if ever.
    -   Potentially low sign-off rate if quality thresholds are hard to meet.

---

## 5. No Revisions ([`no_revisions.json`](./configs/no_revisions.json))

-   **Core Idea**: Simulates a workflow where trainers do not prioritize or perform revisions, effectively meaning tasks are either accepted as is or rejected/stuck.
-   **Key Parameter Settings** (refer to [`no_revisions.json`](./configs/no_revisions.json)):
    -   **TrainerConfig (`trainer_config`)**:
        -   `revision_priority`: Set to 0.0. This means trainers will never choose to work on a task that `NEEDS_WORK` if there's an opportunity to create a new task.
        -   (Optionally, `revision_hours` could be set extremely high to make revisions impractical even if selected, but `revision_priority=0.0` is the direct way).
-   **Expected Outcomes**:
    -   **Tasks Accumulate in `NEEDS_WORK`**: If a task is marked `NEEDS_WORK`, it will likely stay in that state indefinitely as trainers will always prefer creating new tasks.
    -   **Sign-off Depends Entirely on Initial Quality**: Only tasks that meet the quality threshold on the first pass will be signed off.
    -   **Potentially Misleading Throughput**: The number of *new* tasks created might seem reasonable, but the actual number of *useful, signed-off* tasks could be very low if initial quality isn't high enough.

---

## 6. High Variance ([`high_variance.json`](./configs/high_variance.json))

-   **Core Idea**: Models a team or environment where there's a wide range of skill, effort, or task difficulty, leading to highly variable outcomes.
-   **Key Parameter Settings** (affecting both Trainers and Reviewers):
    -   **TrainerConfig (`trainer_config`)**:
        -   `target_hours_per_day_noise`: High value.
        -   `writing_hours_noise`: High value.
        -   `revision_hours_noise`: High value.
        -   `average_initial_quality_noise`: High value (e.g., 0.3-0.4, so quality can swing from very low to very high).
        -   `revision_improvement_noise`: High value.
    -   **ReviewerConfig (`reviewer_config`)**:
        -   `target_hours_per_day_noise`: High value.
        -   `review_hours_noise`: High value.
        -   `quality_threshold_noise`: High value (some reviewers are strict, some lax).
-   **Expected Outcomes**:
    -   **Unpredictable Performance**: Daily/weekly output and quality will fluctuate significantly.
    -   **Wide Distribution of Agent Performance**: Some agents will be highly productive/produce high quality, while others will struggle (visible in agent performance summaries and histograms).
    -   **Mixed Task Outcomes**: Some tasks might get through quickly with high quality, others might get stuck or signed off with low quality due to the variance in reviewer thresholds.
    -   Bottlenecks might appear and disappear erratically.

---

*The actual `.json` configuration files for these scenarios are located in the `configs/` directory. You can inspect them to see the precise parameter settings used.* 