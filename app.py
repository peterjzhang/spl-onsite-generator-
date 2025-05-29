import streamlit as st
import pandas as pd
import numpy as np
import json  # For parameter import/export
import os  # For file system operations
import re  # For filename cleaning
import altair as alt  # Added for better histograms
import datetime  # Added for timestamped filenames
from task_simulator import (
    TaskStatus,
    TrainerConfig,
    ReviewerConfig,
    SimulationConfig,
    DomainSimulationSetup,
    Simulation,
)
from typing import List

# import matplotlib.pyplot as plt # Not strictly necessary for basic st charts but good for custom ones

CONFIG_DIR = "configs"


def create_config_dir():
    os.makedirs(CONFIG_DIR, exist_ok=True)


create_config_dir()  # Ensure directory exists on script start

if (
    "last_uploaded_file_id" not in st.session_state
):  # Initialize tracker for uploaded file
    st.session_state.last_uploaded_file_id = None

# Define known domains, their rates, and an optional initial preset suggestion
PREDEFINED_DOMAINS_SETUP = {
    "Engineering": {"rate": 90.0, "initial_preset_filename": None},
    "Law": {"rate": 90.0, "initial_preset_filename": None},
    "Linguistics": {"rate": 50.0, "initial_preset_filename": None},
    "Business": {"rate": 90.0, "initial_preset_filename": None},
    "Generalist": {"rate": 70.0, "initial_preset_filename": None},
}


def pretty_name_to_filename(pretty_name: str) -> str:
    """Converts a pretty name to a safe filename."""
    s = pretty_name.lower()
    s = re.sub(r"\s+", "_", s)  # Replace spaces with underscores
    s = re.sub(r"[^a-z0-9_\-]", "", s)  # Remove invalid characters
    return s


def filename_to_pretty_name(filename: str) -> str:
    """Converts a filename (without .json) back to a pretty name."""
    s = filename.replace("_", " ").replace("-", " ")
    return s.title()


def generate_download_filename(base_name: str, extension: str, session_state) -> str:
    """Generates a download filename with optional scenario prefix and timestamp."""
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_sim_name_for_file = session_state.get("current_simulation_name", "").strip()
    sluggified_name = (
        pretty_name_to_filename(current_sim_name_for_file)
        if current_sim_name_for_file
        else ""
    )

    if sluggified_name:
        return f"{sluggified_name}_{base_name}_{timestamp_str}.{extension}"
    else:
        return f"{base_name}_{timestamp_str}.{extension}"


def populate_session_state_from_config_dict(config_data, s_state, prefix=""):
    """Populates st.session_state from a loaded configuration dictionary.
    The prefix is used for single config mode directly into session_state.
    """
    sim_settings = config_data.get("simulation_settings", {})
    s_state[prefix + "num_trainers"] = sim_settings.get(
        "num_trainers", s_state.get(prefix + "num_trainers")
    )
    s_state[prefix + "num_reviewers"] = sim_settings.get(
        "num_reviewers", s_state.get(prefix + "num_reviewers")
    )
    s_state[prefix + "simulation_days"] = sim_settings.get(
        "simulation_days", s_state.get(prefix + "simulation_days")
    )

    tc = config_data.get("trainer_config", {})
    s_state[prefix + "trainer_max_hours_per_week"] = tc.get(
        "max_hours_per_week", s_state.get(prefix + "trainer_max_hours_per_week")
    )
    s_state[prefix + "trainer_target_hours_per_day"] = tc.get(
        "target_hours_per_day", s_state.get(prefix + "trainer_target_hours_per_day")
    )
    s_state[prefix + "trainer_target_hours_per_day_noise"] = tc.get(
        "target_hours_per_day_noise",
        s_state.get(prefix + "trainer_target_hours_per_day_noise"),
    )
    s_state[prefix + "trainer_writing_hours"] = tc.get(
        "writing_hours", s_state.get(prefix + "trainer_writing_hours")
    )
    s_state[prefix + "trainer_writing_hours_noise"] = tc.get(
        "writing_hours_noise", s_state.get(prefix + "trainer_writing_hours_noise")
    )
    s_state[prefix + "trainer_revision_hours"] = tc.get(
        "revision_hours", s_state.get(prefix + "trainer_revision_hours")
    )
    s_state[prefix + "trainer_revision_hours_noise"] = tc.get(
        "revision_hours_noise", s_state.get(prefix + "trainer_revision_hours_noise")
    )
    s_state[prefix + "trainer_avg_initial_quality"] = tc.get(
        "average_initial_quality", s_state.get(prefix + "trainer_avg_initial_quality")
    )
    s_state[prefix + "trainer_avg_initial_quality_noise"] = tc.get(
        "average_initial_quality_noise",
        s_state.get(prefix + "trainer_avg_initial_quality_noise"),
    )
    s_state[prefix + "trainer_revision_improvement"] = tc.get(
        "revision_improvement", s_state.get(prefix + "trainer_revision_improvement")
    )
    s_state[prefix + "trainer_revision_improvement_noise"] = tc.get(
        "revision_improvement_noise",
        s_state.get(prefix + "trainer_revision_improvement_noise"),
    )
    s_state[prefix + "trainer_revision_priority"] = tc.get(
        "revision_priority", s_state.get(prefix + "trainer_revision_priority")
    )

    rc = config_data.get("reviewer_config", {})
    s_state[prefix + "reviewer_max_hours_per_week"] = rc.get(
        "max_hours_per_week", s_state.get(prefix + "reviewer_max_hours_per_week")
    )
    s_state[prefix + "reviewer_target_hours_per_day"] = rc.get(
        "target_hours_per_day", s_state.get(prefix + "reviewer_target_hours_per_day")
    )
    s_state[prefix + "reviewer_target_hours_per_day_noise"] = rc.get(
        "target_hours_per_day_noise",
        s_state.get(prefix + "reviewer_target_hours_per_day_noise"),
    )
    s_state[prefix + "reviewer_review_hours"] = rc.get(
        "review_hours", s_state.get(prefix + "reviewer_review_hours")
    )
    s_state[prefix + "reviewer_review_hours_noise"] = rc.get(
        "review_hours_noise", s_state.get(prefix + "reviewer_review_hours_noise")
    )
    s_state[prefix + "reviewer_review_time_decay"] = rc.get(
        "review_time_decay", s_state.get(prefix + "reviewer_review_time_decay")
    )
    s_state[prefix + "reviewer_quality_threshold"] = rc.get(
        "quality_threshold", s_state.get(prefix + "reviewer_quality_threshold")
    )
    s_state[prefix + "reviewer_quality_threshold_noise"] = rc.get(
        "quality_threshold_noise",
        s_state.get(prefix + "reviewer_quality_threshold_noise"),
    )

    # Handle random_seed for single config mode (stored as str in session_state for UI)
    if not prefix:  # Only for single config mode direct session state
        loaded_seed = sim_settings.get("random_seed")  # Can be None or int
        s_state["random_seed_str"] = str(loaded_seed) if loaded_seed is not None else ""


def get_config_from_session_state(s_state, prefix="") -> dict:
    """Gets config dict from st.session_state, using an optional prefix."""
    # Convert random_seed_str to int or None for the actual config dict
    random_seed_val = None
    if not prefix:  # Only for single config mode direct session state
        seed_str = s_state.get("random_seed_str", "").strip()
        if seed_str:
            try:
                random_seed_val = int(seed_str)
            except ValueError:
                # Should ideally show a warning in UI if invalid, but for now pass None
                pass

    return {
        "simulation_settings": {
            "num_trainers": s_state.get(prefix + "num_trainers"),
            "num_reviewers": s_state.get(prefix + "num_reviewers"),
            "simulation_days": s_state.get(prefix + "simulation_days"),
            "random_seed": (
                random_seed_val if not prefix else s_state.get(prefix + "random_seed")
            ),  # Handle appropriately
        },
        "trainer_config": {
            "max_hours_per_week": s_state.get(prefix + "trainer_max_hours_per_week"),
            "target_hours_per_day": s_state.get(
                prefix + "trainer_target_hours_per_day"
            ),
            "target_hours_per_day_noise": s_state.get(
                prefix + "trainer_target_hours_per_day_noise"
            ),
            "writing_hours": s_state.get(prefix + "trainer_writing_hours"),
            "writing_hours_noise": s_state.get(prefix + "trainer_writing_hours_noise"),
            "revision_hours": s_state.get(prefix + "trainer_revision_hours"),
            "revision_hours_noise": s_state.get(
                prefix + "trainer_revision_hours_noise"
            ),
            "average_initial_quality": s_state.get(
                prefix + "trainer_avg_initial_quality"
            ),
            "average_initial_quality_noise": s_state.get(
                prefix + "trainer_avg_initial_quality_noise"
            ),
            "revision_improvement": s_state.get(
                prefix + "trainer_revision_improvement"
            ),
            "revision_improvement_noise": s_state.get(
                prefix + "trainer_revision_improvement_noise"
            ),
            "revision_priority": s_state.get(prefix + "trainer_revision_priority"),
        },
        "reviewer_config": {
            "max_hours_per_week": s_state.get(prefix + "reviewer_max_hours_per_week"),
            "target_hours_per_day": s_state.get(
                prefix + "reviewer_target_hours_per_day"
            ),
            "target_hours_per_day_noise": s_state.get(
                prefix + "reviewer_target_hours_per_day_noise"
            ),
            "review_hours": s_state.get(prefix + "reviewer_review_hours"),
            "review_hours_noise": s_state.get(prefix + "reviewer_review_hours_noise"),
            "review_time_decay": s_state.get(prefix + "reviewer_review_time_decay"),
            "quality_threshold": s_state.get(prefix + "reviewer_quality_threshold"),
            "quality_threshold_noise": s_state.get(
                prefix + "reviewer_quality_threshold_noise"
            ),
        },
    }


def list_preset_configs_from_dir() -> dict:
    """Lists available .json preset configs from the CONFIG_DIR.
    Returns a dict of {pretty_name: filename.json}
    """
    presets = {}
    if os.path.exists(CONFIG_DIR):
        for f_name in os.listdir(CONFIG_DIR):
            if f_name.endswith(".json"):
                base_name = f_name[:-5]  # Remove .json
                presets[filename_to_pretty_name(base_name)] = f_name
    return presets


st.set_page_config(layout="wide")

st.title("Task Creation System Simulator")


# Initialize session state for parameters if they don't exist
def init_parameter_session_state():
    # Single config mode defaults
    defaults = {
        "simulation_mode": "Single Configuration",
        "num_trainers": 60,
        "num_reviewers": 40,
        "simulation_days": 21,
        "trainer_max_hours_per_week": 40.0,
        "trainer_target_hours_per_day": 5.0,
        "trainer_target_hours_per_day_noise": 0.3,
        "trainer_writing_hours": 5.0,
        "trainer_writing_hours_noise": 0.3,
        "trainer_revision_hours": 3.0,
        "trainer_revision_hours_noise": 0.4,
        "trainer_avg_initial_quality": 0.5,
        "trainer_avg_initial_quality_noise": 0.2,
        "trainer_revision_improvement": 0.1,
        "trainer_revision_improvement_noise": 0.5,
        "trainer_revision_priority": 0.7,
        "reviewer_max_hours_per_week": 40.0,
        "reviewer_target_hours_per_day": 5.0,
        "reviewer_target_hours_per_day_noise": 0.25,
        "reviewer_review_hours": 5.0,
        "reviewer_review_hours_noise": 0.35,
        "reviewer_review_time_decay": 0.9,
        "reviewer_quality_threshold": 0.90,
        "reviewer_quality_threshold_noise": 0.05,
        "current_simulation_name": "My Scenario",
        "random_seed_str": "",  # For UI text input, store seed as string
        # Multi-domain state
        "domain_configs": {},
        "active_domains": [],
        "final_sim_config": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_parameter_session_state()

# --- Sidebar for Parameters ---
st.sidebar.header("Simulation Parameters")

st.session_state.simulation_mode = st.sidebar.radio(
    "Simulation Mode",
    ("Single Configuration", "Multi-Domain Configuration"),
    key="sim_mode_radio",
)

# --- Parameter Management (Common to both modes for now, might split later) ---
st.sidebar.subheader("Configuration Management")

st.session_state.current_simulation_name = st.sidebar.text_input(
    "Current Scenario Name (Optional for Single Mode, Used for Saving Presets):",
    value=st.session_state.get("current_simulation_name", ""),
    key="s_current_simulation_name_input",
)

# --- Load/Save Presets (Global Presets) ---
available_presets = list_preset_configs_from_dir()

if st.session_state.simulation_mode == "Single Configuration":
    st.sidebar.markdown("**Load Preset for Single Config**")
    if available_presets:
        selected_preset_pretty_name_single = st.sidebar.selectbox(
            "Choose a preset to load for this configuration:",
            options=["None"] + list(available_presets.keys()),
            key="select_preset_single",
        )
        if (
            selected_preset_pretty_name_single
            and selected_preset_pretty_name_single != "None"
        ):
            if st.sidebar.button(
                "Load Selected Preset into Current Single Config",
                key="load_preset_button_single",
            ):
                try:
                    file_to_load = available_presets[selected_preset_pretty_name_single]
                    with open(os.path.join(CONFIG_DIR, file_to_load), "r") as f:
                        config_data = json.load(f)
                        # Populate main session state for single config mode
                        populate_session_state_from_config_dict(
                            config_data, st.session_state
                        )
                        # Also update current_simulation_name if the preset name is different
                        if (
                            filename_to_pretty_name(file_to_load[:-5])
                            != st.session_state.current_simulation_name
                        ):
                            st.session_state.current_simulation_name = (
                                filename_to_pretty_name(file_to_load[:-5])
                            )
                        st.sidebar.success(
                            f"Preset '{selected_preset_pretty_name_single}' loaded!"
                        )
                        st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error loading preset: {e}")
    else:
        st.sidebar.caption("No saved global presets found to load.")

st.sidebar.markdown("**Save Current Single Config Settings as Global Preset**")
if st.sidebar.button(
    "Save Current Single Config as Preset", key="save_preset_button_single"
):
    sim_name_to_save = st.session_state.get("current_simulation_name", "").strip()
    if not sim_name_to_save:
        st.sidebar.warning("Please enter a 'Current Scenario Name' to save the preset.")
    else:
        filename_base = pretty_name_to_filename(sim_name_to_save)
        filename = f"{filename_base}.json"
        file_path = os.path.join(CONFIG_DIR, filename)
        # Get config from main session state for single config mode
        current_config_to_save = get_config_from_session_state(st.session_state)
        try:
            with open(file_path, "w") as f:
                json.dump(current_config_to_save, f, indent=4)
            st.sidebar.success(f"Preset '{sim_name_to_save}' saved as {filename}!")
        except Exception as e:
            st.sidebar.error(f"Error saving preset: {e}")

# Ad-hoc upload for single config (could be adapted for multi-domain later if needed)
st.sidebar.markdown("--- Ad-hoc Import/Export (Current Single Config) ---")
uploaded_config_file = st.sidebar.file_uploader(
    "Upload Configuration JSON", type=["json"], key="config_uploader_single"
)

if uploaded_config_file is not None:
    current_file_id = (
        uploaded_config_file.file_id
    )  # Use .file_id for a unique identifier
    if current_file_id != st.session_state.get("last_uploaded_file_id"):
        try:
            # Make sure to reset the file pointer before reading, if it might have been read before
            uploaded_config_file.seek(0)
            config_data = json.load(uploaded_config_file)
            populate_session_state_from_config_dict(config_data, st.session_state)

            st.session_state.last_uploaded_file_id = (
                current_file_id  # Mark this file ID as processed
            )
            st.sidebar.success("Uploaded configuration loaded!")
        except json.JSONDecodeError:
            st.sidebar.error("Invalid JSON file in uploader.")
            st.session_state.last_uploaded_file_id = (
                None  # Allow re-processing if error on this file
            )
        except Exception as e:
            st.sidebar.error(f"Error loading uploaded config: {e}")
            st.session_state.last_uploaded_file_id = (
                None  # Allow re-processing if error on this file
            )

current_config_dict = get_config_from_session_state(st.session_state)
json_export_string = json.dumps(current_config_dict, indent=4)

# Determine default filename for config export using the new helper
config_export_filename = generate_download_filename(
    "simulation_config", "json", st.session_state
)

st.sidebar.download_button(
    label="Download Current Configuration JSON",
    data=json_export_string,
    file_name=config_export_filename,
    mime="application/json",
)

# --- UI for Single Configuration Mode ---
if st.session_state.simulation_mode == "Single Configuration":
    st.sidebar.subheader("General Simulation Settings (Single Config)")
    st.session_state.num_trainers = st.sidebar.slider(
        "Number of Trainers",
        1,
        100,
        st.session_state.num_trainers,
        key="s_num_trainers_single",
    )
    st.session_state.num_reviewers = st.sidebar.slider(
        "Number of Reviewers",
        1,
        100,
        st.session_state.num_reviewers,
        key="s_num_reviewers_single",
    )
    st.session_state.simulation_days = st.sidebar.slider(
        "Simulation Duration (Days)",
        7,
        90,
        st.session_state.simulation_days,
        7,
        key="s_simulation_days_single",
    )

    st.sidebar.subheader("Trainer Agent Configuration (Single Config)")
    st.session_state.trainer_max_hours_per_week = st.sidebar.slider(
        "Trainer: Max Hours/Week",
        10.0,
        80.0,
        st.session_state.trainer_max_hours_per_week,
        0.5,
        key="s_trainer_max_hours_week_single",
    )
    st.session_state.trainer_target_hours_per_day = st.sidebar.slider(
        "Trainer: Target Hours/Day (Mean)",
        1.0,
        12.0,
        st.session_state.trainer_target_hours_per_day,
        0.5,
        key="s_trainer_target_hours_day_single",
    )
    st.session_state.trainer_target_hours_per_day_noise = st.sidebar.number_input(
        "└ Target Hours/Day (Coeff. of Variation)",
        min_value=0.01,
        max_value=2.0,
        value=st.session_state.trainer_target_hours_per_day_noise,
        step=0.05,
        format="%.2f",
        key="s_trainer_target_hours_noise_single",
    )
    st.session_state.trainer_writing_hours = st.sidebar.slider(
        "Trainer: Writing Hours/Task (Mean)",
        1.0,
        20.0,
        st.session_state.trainer_writing_hours,
        0.5,
        key="s_trainer_writing_hours_single",
    )
    st.session_state.trainer_writing_hours_noise = st.sidebar.number_input(
        "└ Writing Hours/Task (Coeff. of Variation)",
        min_value=0.01,
        max_value=2.0,
        value=st.session_state.trainer_writing_hours_noise,
        step=0.05,
        format="%.2f",
        key="s_trainer_writing_hours_noise_single",
    )
    st.session_state.trainer_revision_hours = st.sidebar.slider(
        "Trainer: Revision Hours/Task (Mean)",
        0.5,
        10.0,
        st.session_state.trainer_revision_hours,
        0.25,
        key="s_trainer_revision_hours_single",
    )
    st.session_state.trainer_revision_hours_noise = st.sidebar.number_input(
        "└ Revision Hours/Task (Coeff. of Variation)",
        min_value=0.01,
        max_value=2.0,
        value=st.session_state.trainer_revision_hours_noise,
        step=0.05,
        format="%.2f",
        key="s_trainer_revision_hours_noise_single",
    )
    st.session_state.trainer_avg_initial_quality = st.sidebar.slider(
        "Trainer: Average Initial Quality (Normal Mean)",
        0.0,
        1.0,
        st.session_state.trainer_avg_initial_quality,
        0.01,
        key="s_trainer_avg_initial_quality_single",
    )
    st.session_state.trainer_avg_initial_quality_noise = st.sidebar.number_input(
        "└ Avg Initial Quality Noise (Normal StdDev)",
        min_value=0.0,
        max_value=0.5,
        value=st.session_state.trainer_avg_initial_quality_noise,
        step=0.01,
        format="%.2f",
        key="s_trainer_avg_initial_quality_noise_single",
    )
    st.session_state.trainer_revision_improvement = st.sidebar.slider(
        "Trainer: Revision Quality Improvement (Mean)",
        0.0,
        0.5,
        st.session_state.trainer_revision_improvement,
        0.01,
        key="s_trainer_revision_improvement_single",
    )
    st.session_state.trainer_revision_improvement_noise = st.sidebar.number_input(
        "└ Revision Improvement (Coeff. of Variation)",
        min_value=0.01,
        max_value=2.0,
        value=st.session_state.trainer_revision_improvement_noise,
        step=0.05,
        format="%.2f",
        key="s_trainer_revision_improvement_noise_single",
    )
    st.session_state.trainer_revision_priority = st.sidebar.slider(
        "Trainer: Revision Priority (Prob)",
        0.0,
        1.0,
        st.session_state.trainer_revision_priority,
        0.05,
        key="s_trainer_revision_priority_single",
    )

    st.sidebar.subheader("Reviewer Agent Configuration (Single Config)")
    st.session_state.reviewer_max_hours_per_week = st.sidebar.slider(
        "Reviewer: Max Hours/Week",
        20.0,
        80.0,
        st.session_state.reviewer_max_hours_per_week,
        1.0,
        key="s_reviewer_max_hours_week_single",
    )
    st.session_state.reviewer_target_hours_per_day = st.sidebar.slider(
        "Reviewer: Target Hours/Day (Mean)",
        1.0,
        12.0,
        st.session_state.reviewer_target_hours_per_day,
        0.5,
        key="s_reviewer_target_hours_day_single",
    )
    st.session_state.reviewer_target_hours_per_day_noise = st.sidebar.number_input(
        "└ Target Hours/Day (Coeff. of Variation)",
        min_value=0.01,
        max_value=2.0,
        value=st.session_state.reviewer_target_hours_per_day_noise,
        step=0.05,
        format="%.2f",
        key="s_reviewer_target_hours_noise_single",
    )
    st.session_state.reviewer_review_hours = st.sidebar.slider(
        "Reviewer: Review Hours/Task (Mean)",
        0.5,
        10.0,
        st.session_state.reviewer_review_hours,
        0.25,
        key="s_reviewer_review_hours_single",
    )
    st.session_state.reviewer_review_hours_noise = st.sidebar.number_input(
        "└ Review Hours/Task (Coeff. of Variation)",
        min_value=0.01,
        max_value=2.0,
        value=st.session_state.reviewer_review_hours_noise,
        step=0.05,
        format="%.2f",
        key="s_reviewer_review_hours_noise_single",
    )
    st.session_state.reviewer_review_time_decay = st.sidebar.slider(
        "Reviewer: Review Time Decay Factor",
        0.5,
        1.0,
        st.session_state.get("reviewer_review_time_decay", 0.9),
        0.05,
        key="s_reviewer_review_time_decay_single",
        help="Efficiency gain when reviewing the same person repeatedly. 0.9 = 10% faster each time.",
    )
    st.session_state.reviewer_quality_threshold = st.sidebar.slider(
        "Reviewer: Quality Threshold for Sign-off (Normal Mean)",
        0.0,
        1.0,
        st.session_state.reviewer_quality_threshold,
        0.01,
        key="s_reviewer_quality_threshold_single",
    )
    st.session_state.reviewer_quality_threshold_noise = st.sidebar.number_input(
        "└ Quality Threshold Noise (Normal StdDev)",
        min_value=0.0,
        max_value=0.5,
        value=st.session_state.reviewer_quality_threshold_noise,
        step=0.01,
        format="%.2f",
        key="s_reviewer_quality_threshold_noise_single",
    )

# --- UI for Multi-Domain Configuration Mode ---
elif st.session_state.simulation_mode == "Multi-Domain Configuration":
    st.sidebar.subheader("Multi-Domain Setup")
    st.sidebar.markdown(
        "Define agent groups for different domains. Each domain uses a global preset for detailed agent characteristics, including number of trainers/reviewers."
    )

    # Initialize active_domains from PREDEFINED_DOMAINS_SETUP if empty
    if not st.session_state.active_domains:
        st.session_state.active_domains = list(PREDEFINED_DOMAINS_SETUP.keys())
        for domain_key in st.session_state.active_domains:
            if domain_key not in st.session_state.domain_configs:
                domain_defaults = PREDEFINED_DOMAINS_SETUP[domain_key]
                st.session_state.domain_configs[domain_key] = {
                    "display_name": domain_key,
                    "preset_filename": domain_defaults["initial_preset_filename"],
                    "rate": domain_defaults["rate"],
                    "loaded_num_trainers": 0,
                    "loaded_num_reviewers": 0,
                    "scale_factor": 1.0,  # Added scale factor
                }

    for domain_key in list(st.session_state.active_domains):
        domain_config_entry = st.session_state.domain_configs[domain_key]
        with st.sidebar.expander(
            f"Domain Settings: {domain_config_entry['display_name']}", expanded=True
        ):
            new_display_name = st.text_input(
                "Domain Name (Editable)",
                value=domain_config_entry["display_name"],
                key=f"domain_display_name_{domain_key}",
            )
            if new_display_name != domain_config_entry["display_name"]:
                st.session_state.domain_configs[domain_key][
                    "display_name"
                ] = new_display_name

            # Add Scale Factor input
            current_scale_factor = domain_config_entry.get("scale_factor", 1.0)
            new_scale_factor = st.number_input(
                "Scale Factor (for agent counts)",
                min_value=0.1,
                max_value=10.0,
                value=current_scale_factor,
                step=0.1,
                format="%.1f",
                key=f"scale_factor_{domain_key}",
            )
            if new_scale_factor != current_scale_factor:
                st.session_state.domain_configs[domain_key][
                    "scale_factor"
                ] = new_scale_factor
                # Rerun to update displayed scaled agent counts immediately
                st.rerun()

            current_preset_file = domain_config_entry.get("preset_filename")
            current_preset_pretty = (
                filename_to_pretty_name(current_preset_file[:-5])
                if current_preset_file
                else "None"
            )

            preset_options = ["None"] + list(available_presets.keys())
            selected_preset_for_domain = st.selectbox(
                f"Agent Config Preset for {domain_config_entry['display_name']}",
                options=preset_options,
                index=(
                    preset_options.index(current_preset_pretty)
                    if current_preset_pretty in preset_options
                    else 0
                ),
                key=f"preset_select_{domain_key}",
            )

            # Logic to update preset_filename and try to load agent counts for display
            if selected_preset_for_domain != current_preset_pretty:
                new_preset_filename = None
                if selected_preset_for_domain != "None":
                    new_preset_filename = available_presets[selected_preset_for_domain]

                st.session_state.domain_configs[domain_key][
                    "preset_filename"
                ] = new_preset_filename
                # Attempt to load num_trainers/reviewers from the new preset for display purposes
                if new_preset_filename:
                    try:
                        with open(
                            os.path.join(CONFIG_DIR, new_preset_filename), "r"
                        ) as f:
                            preset_data = json.load(f)
                        sim_settings = preset_data.get("simulation_settings", {})
                        st.session_state.domain_configs[domain_key][
                            "loaded_num_trainers"
                        ] = sim_settings.get("num_trainers", 0)
                        st.session_state.domain_configs[domain_key][
                            "loaded_num_reviewers"
                        ] = sim_settings.get("num_reviewers", 0)
                    except Exception as e:
                        st.warning(
                            f"Could not read agent counts from {new_preset_filename}: {e}"
                        )
                        st.session_state.domain_configs[domain_key][
                            "loaded_num_trainers"
                        ] = 0
                        st.session_state.domain_configs[domain_key][
                            "loaded_num_reviewers"
                        ] = 0
                else:
                    st.session_state.domain_configs[domain_key][
                        "loaded_num_trainers"
                    ] = 0
                    st.session_state.domain_configs[domain_key][
                        "loaded_num_reviewers"
                    ] = 0
                st.rerun()

            # Display the number of trainers/reviewers from the loaded preset, applying the scale factor
            if domain_config_entry.get("preset_filename"):
                loaded_trainers_base = domain_config_entry.get("loaded_num_trainers", 0)
                loaded_reviewers_base = domain_config_entry.get(
                    "loaded_num_reviewers", 0
                )
                scale = domain_config_entry.get("scale_factor", 1.0)

                st.markdown(
                    f"_Base Trainers (from preset): **{loaded_trainers_base}**_"
                )
                st.markdown(
                    f"_Base Reviewers (from preset): **{loaded_reviewers_base}**_"
                )

                if scale != 1.0:
                    scaled_trainers = max(0, int(loaded_trainers_base * scale))
                    scaled_reviewers = max(0, int(loaded_reviewers_base * scale))
                    st.markdown(
                        f"_**Scaled Trainers for Sim: {scaled_trainers}**_ ({scale:.1f}x)"
                    )
                    st.markdown(
                        f"_**Scaled Reviewers for Sim: {scaled_reviewers}**_ ({scale:.1f}x)"
                    )
            else:
                st.markdown(
                    "_Select a preset to define base agent counts and configs._"
                )

    st.sidebar.markdown("--- Simulation Days (Multi-Domain) ---")
    st.session_state.simulation_days = st.sidebar.slider(
        "Overall Simulation Duration (Days)",
        7,
        90,
        st.session_state.get(
            "simulation_days", 21
        ),  # Use general simulation_days from session_state
        7,
        key="s_simulation_days_multi",
    )

# --- Run Simulation Section ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎲 Run Simulation")

# Random seed input (common to both modes, right before the run button)
st.session_state.random_seed_str = st.sidebar.text_input(
    "Random Seed (Optional, for deterministic runs):",
    value=st.session_state.get("random_seed_str", ""),
    key="s_random_seed_input",
    help="Enter an integer to make simulation results reproducible. Leave empty for random behavior.",
).strip()

# Display current seed status
if st.session_state.random_seed_str:
    try:
        int(st.session_state.random_seed_str)
        st.sidebar.success(f"🔒 Will run with seed: {st.session_state.random_seed_str}")
    except ValueError:
        st.sidebar.error("❌ Invalid seed - must be an integer")
else:
    st.sidebar.info("🎯 Random seed not set - results will vary between runs")

# --- Simulation Execution & Display ---
if st.sidebar.button("Run Simulation", key="run_sim_button"):
    st.session_state.simulation_run = False
    st.session_state.final_sim_config = None  # Reset on new run attempt

    domain_setups_for_sim: List[DomainSimulationSetup] = []
    sim_days: int = st.session_state.get("simulation_days", 21)
    # sim_cfg_object: Optional[SimulationConfig] = None # No longer needed as local var for this purpose

    if st.session_state.simulation_mode == "Single Configuration":
        # Create a single DomainSimulationSetup from the main session state parameters
        single_trainer_cfg = TrainerConfig(
            max_hours_per_week=st.session_state.trainer_max_hours_per_week,
            target_hours_per_day=st.session_state.trainer_target_hours_per_day,
            target_hours_per_day_noise=st.session_state.trainer_target_hours_per_day_noise,
            writing_hours=st.session_state.trainer_writing_hours,
            writing_hours_noise=st.session_state.trainer_writing_hours_noise,
            revision_hours=st.session_state.trainer_revision_hours,
            revision_hours_noise=st.session_state.trainer_revision_hours_noise,
            average_initial_quality=st.session_state.trainer_avg_initial_quality,
            average_initial_quality_noise=st.session_state.trainer_avg_initial_quality_noise,
            revision_improvement=st.session_state.trainer_revision_improvement,
            revision_improvement_noise=st.session_state.trainer_revision_improvement_noise,
            revision_priority=st.session_state.trainer_revision_priority,
        )
        single_reviewer_cfg = ReviewerConfig(
            max_hours_per_week=st.session_state.reviewer_max_hours_per_week,
            target_hours_per_day=st.session_state.reviewer_target_hours_per_day,
            target_hours_per_day_noise=st.session_state.reviewer_target_hours_per_day_noise,
            review_hours=st.session_state.reviewer_review_hours,
            review_hours_noise=st.session_state.reviewer_review_hours_noise,
            review_time_decay=st.session_state.reviewer_review_time_decay,
            quality_threshold=st.session_state.reviewer_quality_threshold,
            quality_threshold_noise=st.session_state.reviewer_quality_threshold_noise,
        )
        domain_name_single = (
            st.session_state.get("current_simulation_name", "Default").strip()
            or "Default"
        )
        domain_setups_for_sim.append(
            DomainSimulationSetup(
                domain_name=domain_name_single,
                num_trainers=st.session_state.num_trainers,
                num_reviewers=st.session_state.num_reviewers,
                trainer_cfg=single_trainer_cfg,
                reviewer_cfg=single_reviewer_cfg,
            )
        )

    elif st.session_state.simulation_mode == "Multi-Domain Configuration":
        any_domain_misconfigured = False
        for domain_key in st.session_state.active_domains:
            domain_detail = st.session_state.domain_configs[domain_key]
            preset_filename_for_domain = domain_detail.get("preset_filename")

            if not preset_filename_for_domain:
                st.sidebar.error(
                    f"Error: Domain '{domain_key}' is missing a selected agent config preset."
                )
                any_domain_misconfigured = True
                continue

            try:
                num_trainers_from_preset = 0
                num_reviewers_from_preset = 0
                with open(
                    os.path.join(CONFIG_DIR, preset_filename_for_domain), "r"
                ) as f:
                    preset_data = json.load(f)

                sim_settings = preset_data.get("simulation_settings", {})
                num_trainers_from_preset = sim_settings.get("num_trainers", 0)
                num_reviewers_from_preset = sim_settings.get("num_reviewers", 0)

                # Update loaded base counts in session state (these are pre-scale)
                st.session_state.domain_configs[domain_key][
                    "loaded_num_trainers"
                ] = num_trainers_from_preset
                st.session_state.domain_configs[domain_key][
                    "loaded_num_reviewers"
                ] = num_reviewers_from_preset

                # Apply scale factor for the actual simulation setup
                scale_factor = domain_detail.get(
                    "scale_factor", 1.0
                )  # Get scale factor from domain_detail
                actual_num_trainers = max(
                    0, int(num_trainers_from_preset * scale_factor)
                )
                actual_num_reviewers = max(
                    0, int(num_reviewers_from_preset * scale_factor)
                )

                if actual_num_trainers == 0 and actual_num_reviewers == 0:
                    st.sidebar.warning(
                        f"Skipping domain '{domain_detail['display_name']}' as its selected preset ('{preset_filename_for_domain}') and scale factor ({scale_factor}x) results in 0 trainers and 0 reviewers."
                    )
                    continue

                loaded_trainer_cfg_data = preset_data.get("trainer_config", {})
                loaded_reviewer_cfg_data = preset_data.get("reviewer_config", {})

                domain_trainer_cfg = TrainerConfig(**loaded_trainer_cfg_data)
                domain_reviewer_cfg = ReviewerConfig(**loaded_reviewer_cfg_data)

                domain_setups_for_sim.append(
                    DomainSimulationSetup(
                        domain_name=domain_detail["display_name"],
                        num_trainers=actual_num_trainers,  # Use scaled count
                        num_reviewers=actual_num_reviewers,  # Use scaled count
                        trainer_cfg=domain_trainer_cfg,
                        reviewer_cfg=domain_reviewer_cfg,
                    )
                )
            except FileNotFoundError:
                st.sidebar.error(
                    f"Preset file {preset_filename_for_domain} not found for domain {domain_key}."
                )
                any_domain_misconfigured = True
            except Exception as e:
                st.sidebar.error(f"Error loading preset for domain {domain_key}: {e}")
                any_domain_misconfigured = True

        if any_domain_misconfigured or not domain_setups_for_sim:
            st.sidebar.error(
                "Multi-domain simulation cannot run due to configuration errors or no valid domains."
            )
            st.session_state.simulation_run = False
        else:
            sim_days = (
                st.session_state.simulation_days
            )  # Ensure sim_days is set for multi-domain

    # Get random seed from session state (string) and convert to int/None
    current_random_seed = None
    seed_input_str = st.session_state.get("random_seed_str", "").strip()
    if seed_input_str:
        try:
            current_random_seed = int(seed_input_str)
        except ValueError:
            st.sidebar.error(
                f"Invalid Random Seed: '{seed_input_str}'. Must be an integer. Running without a fixed seed."
            )
            # No explicit rerun, will just proceed without seed for this run

    if domain_setups_for_sim:  # Proceed only if there are valid setups
        sim_cfg_object_local = SimulationConfig(
            simulation_days=sim_days,
            domain_setups=domain_setups_for_sim,
            random_seed=current_random_seed,  # Pass the processed seed
        )
        simulation = Simulation(config=sim_cfg_object_local)
        df_summary = simulation.run()
        st.session_state.df_summary = df_summary
        st.session_state.simulation_run = True
        st.session_state.tasks_final_state = simulation.tasks
        st.session_state.sim_trainers = simulation.trainers
        st.session_state.sim_reviewers = simulation.reviewers
        st.session_state.final_sim_config = (
            sim_cfg_object_local  # Store the successful config
        )
    else:
        st.sidebar.warning(
            "No valid simulation setups to run. Please check configurations."
        )

else:
    if "simulation_run" not in st.session_state:
        st.session_state.simulation_run = False

if st.session_state.simulation_run and hasattr(st.session_state, "df_summary"):
    df_summary = st.session_state.df_summary
    st.header("Simulation Results")

    # Display random seed used if any
    final_config = st.session_state.get("final_sim_config")
    if final_config and final_config.random_seed is not None:
        st.success(
            f"🔒 Simulation run with Random Seed: **{final_config.random_seed}** (Results are deterministic)"
        )
    else:
        st.info(
            "🎯 Simulation run without random seed (Results will vary between runs)"
        )

    # Display overall simulation name if set, or mode
    overall_sim_name = st.session_state.get("current_simulation_name", "").strip()
    if st.session_state.simulation_mode == "Multi-Domain Configuration":
        active_domain_names_display = []
        # Check st.session_state for the config from the last successful run
        final_config = st.session_state.get("final_sim_config")
        if (
            final_config is not None
            and hasattr(final_config, "domain_setups")
            and final_config.domain_setups
        ):
            active_domain_names_display = [
                ds.domain_name for ds in final_config.domain_setups
            ]

        if active_domain_names_display:
            st.subheader(
                f"Results for Multi-Domain Simulation: {', '.join(active_domain_names_display)}"
            )
        else:
            st.subheader(f"Results for Multi-Domain Simulation (No valid domains ran)")
    elif overall_sim_name:
        st.subheader(f"Results for Scenario: {overall_sim_name}")

    # Create columns for side-by-side charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Daily Task Flow")
        st.line_chart(
            df_summary.set_index("day")[
                [
                    "new_tasks_created",
                    "tasks_writing_completed",
                    "tasks_revision_completed",
                    "tasks_decisioned_by_review",
                ]
            ]
        )

    with col2:
        st.subheader("Task Queue Status (End of Day)")
        st.line_chart(
            df_summary.set_index("day")[
                [
                    "tasks_needing_work_eod",
                    "tasks_complete_waiting_review_eod",
                    "tasks_fixing_done_waiting_review_eod",
                    # 'total_tasks_in_system' # Consider if this makes the scale too different, optional
                ]
            ]
        )

    # New Stacked Bar Chart for Task Statuses
    st.subheader("Cumulative Task Status Overview (End of Day)")

    # Define the desired order and display names for status columns
    # This also defines the stacking order from bottom to top
    ordered_status_map = {
        "tasks_claimed_eod": "1. Claimed",
        "tasks_needing_work_eod": "2. Needs Work",
        "tasks_complete_waiting_review_eod": "3. Awaiting Review (New)",
        "tasks_fixing_done_waiting_review_eod": "4. Awaiting Review (Revised)",
        "tasks_review_in_progress_eod": "5. In Review",
        "signed_off_cumulative": "6. Signed Off",
    }
    ordered_status_keys = list(ordered_status_map.keys())
    ordered_display_names = [ordered_status_map[key] for key in ordered_status_keys]

    # Colors must correspond to ordered_display_names
    status_colors = [
        "#D3D3D3",  # 1. Claimed (Light Gray)
        "#FF6347",  # 2. Needs Work (Tomato Red)
        "#FFD700",  # 3. Awaiting Review (New) (Gold)
        "#FFA500",  # 4. Awaiting Review (Revised) (Orange)
        "#1E90FF",  # 5. In Review (Dodger Blue)
        "#32CD32",  # 6. Signed Off (Lime Green)
    ]

    # Prepare DataFrame for plotting, ensuring all columns exist and are in order
    df_plot_ready = df_summary[["day"]].copy()
    for original_key, display_name in ordered_status_map.items():
        if original_key in df_summary.columns:
            df_plot_ready[display_name] = df_summary[original_key]
        else:
            df_plot_ready[display_name] = (
                0  # Add column with zeros if missing from source
            )

    df_plot_ready = df_plot_ready.set_index("day")
    # Select only the ordered display name columns for the chart
    df_plot_ready = df_plot_ready[ordered_display_names]

    if not df_plot_ready.empty:
        st.bar_chart(df_plot_ready, color=status_colors)  # type: ignore[arg-type]
        st.caption(
            "This chart shows the cumulative number of tasks in each status at the end of each day."
        )
    else:
        st.write("No data available for task status overview.")

    st.subheader("Quality & Agent Workload")
    col_qual, col_work = st.columns(
        2
    )  # Renamed to avoid conflict with col1, col2 above
    with col_qual:
        st.markdown("**Average Quality of Signed-off Tasks (Cumulative)**")
        # Conditional plotting for average quality
        avg_quality_series = df_summary.set_index("day")["avg_quality_signed_off"]
        # Check if there are any non-zero quality scores, implying some tasks were signed off
        if not avg_quality_series.empty and (avg_quality_series > 0).any():
            st.line_chart(avg_quality_series)
        else:
            st.write("No tasks signed off yet, or average quality is zero.")

    with col_work:
        st.markdown("**Average Hours Worked (Per Day)**")
        st.line_chart(
            df_summary.set_index("day")[
                [
                    "avg_trainer_hrs_worked_today",
                    "avg_reviewer_hrs_worked_today",
                ]  # Removed weekly averages
            ]
        )
        # st.caption("'_week' metrics show average hours worked so far in the current week.") # Caption removed

    st.subheader("Raw Daily Summary Data")
    st.dataframe(df_summary)
    # Add download button for daily summary
    csv_daily_summary = df_summary.to_csv(index=False).encode("utf-8")
    daily_summary_filename = generate_download_filename(
        "daily_simulation_summary", "csv", st.session_state
    )
    st.download_button(
        label="Download Daily Summary as CSV",
        data=csv_daily_summary,
        file_name=daily_summary_filename,
        mime="text/csv",
        key="download_daily_summary",
    )

    if (
        hasattr(st.session_state, "tasks_final_state")
        and st.session_state.tasks_final_state
    ):
        st.subheader("Final State of All Tasks (Global)")
        all_tasks_data = []
        for task in st.session_state.tasks_final_state:
            task_entry = {
                "ID": task.id,
                "Owner ID": task.owner_id,
                "Owner Domain": task.owner_domain,
                "Reviewer ID": task.reviewer_id,
                "Reviewer Domain": task.reviewer_domain,
                "Status": task.status.name,
                "Quality": f"{task.quality_score:.2f}",
                "Revisions": task.revision_count,
                "Review Cycles": task.revision_count + 1,
                "Major Issues": task.major_issues,
                "Minor Issues": task.minor_issues,
                "Total Writing Hours": f"{task.writing_progress_hours:.2f}",
                "Total Revision Hours": f"{task.revision_progress_hours:.2f}",
                "Total Review Hours": f"{task.review_progress_hours:.2f}",
            }
            all_tasks_data.append(task_entry)

        df_all_tasks = pd.DataFrame(all_tasks_data)
        # Define column order for final tasks table
        final_tasks_cols_order = [
            "ID",
            "Status",
            "Owner ID",
            "Owner Domain",
            "Reviewer ID",
            "Reviewer Domain",
            "Quality",
            "Revisions",
            "Review Cycles",
            "Total Writing Hours",
            "Total Revision Hours",
            "Total Review Hours",
            "Major Issues",
            "Minor Issues",
        ]
        # Ensure all columns are present
        for col in final_tasks_cols_order:
            if col not in df_all_tasks.columns:
                df_all_tasks[col] = None  # Or a suitable default like "N/A"
        st.dataframe(df_all_tasks[final_tasks_cols_order])

        # Add download button for final task list
        csv_final_tasks = df_all_tasks.to_csv(index=False).encode("utf-8")
        final_tasks_filename = generate_download_filename(
            "final_task_list", "csv", st.session_state
        )
        st.download_button(
            label="Download Final Task List as CSV",
            data=csv_final_tasks,
            file_name=final_tasks_filename,
            mime="text/csv",
            key="download_final_tasks",
        )

        # Consolidated Agent Performance Summary Table
        st.subheader("Agent Performance Summary (End of Simulation)")
        agent_performance_data = []
        if hasattr(st.session_state, "sim_trainers") and st.session_state.sim_trainers:
            for trainer in st.session_state.sim_trainers:
                tasks_created = sum(
                    1
                    for t in st.session_state.tasks_final_state
                    if t.owner_id == trainer.id
                )
                signed_off_owned_tasks = [
                    t
                    for t in st.session_state.tasks_final_state
                    if t.owner_id == trainer.id and t.status == TaskStatus.SIGNED_OFF
                ]
                tasks_signed_off_owned_count = len(signed_off_owned_tasks)
                avg_major_issues_owned_signed_off = (
                    float(np.mean([t.major_issues for t in signed_off_owned_tasks]))
                    if signed_off_owned_tasks
                    else 0.0
                )
                avg_minor_issues_owned_signed_off = (
                    float(np.mean([t.minor_issues for t in signed_off_owned_tasks]))
                    if signed_off_owned_tasks
                    else 0.0
                )
                avg_quality_owned_signed_off = (
                    float(np.mean([t.quality_score for t in signed_off_owned_tasks]))
                    if signed_off_owned_tasks
                    else 0.0
                )
                avg_review_cycles_owned_signed_off = (
                    float(
                        np.mean([t.revision_count + 1 for t in signed_off_owned_tasks])
                    )
                    if signed_off_owned_tasks
                    else 0.0
                )
                tasks_needing_work_owned = sum(
                    1
                    for t in st.session_state.tasks_final_state
                    if t.owner_id == trainer.id and t.status == TaskStatus.NEEDS_WORK
                )
                agent_performance_data.append(
                    {
                        "Agent ID": trainer.id,
                        "Type": "Trainer",
                        "Domain": trainer.domain_name,
                        "Tasks Created": tasks_created,
                        "Tasks Signed Off (Owned)": tasks_signed_off_owned_count,
                        "Avg Quality (Signed Off Owned)": f"{avg_quality_owned_signed_off:.2f}",
                        "Avg Review Cycles (Signed Off Owned)": f"{avg_review_cycles_owned_signed_off:.1f}",
                        "Avg Major Issues (Signed Off Owned)": f"{avg_major_issues_owned_signed_off:.2f}",
                        "Avg Minor Issues (Signed Off Owned)": f"{avg_minor_issues_owned_signed_off:.2f}",
                        "Tasks Needing Work (Owned)": tasks_needing_work_owned,
                        "Tasks Reviewed": 0,
                        "Tasks Signed Off (Reviewed)": 0,
                        "Avg Quality (Signed Off Reviewed)": f"{0.0:.2f}",
                        "Avg Review Cycles (Signed Off Reviewed)": f"{0.0:.1f}",
                        "Avg Major Issues (Signed Off Reviewed)": f"{0.0:.2f}",
                        "Avg Minor Issues (Signed Off Reviewed)": f"{0.0:.2f}",
                    }
                )

        if (
            hasattr(st.session_state, "sim_reviewers")
            and st.session_state.sim_reviewers
        ):
            for reviewer in st.session_state.sim_reviewers:
                signed_off_reviewed_tasks = [
                    t
                    for t in st.session_state.tasks_final_state
                    if t.reviewer_id == reviewer.id
                    and t.status == TaskStatus.SIGNED_OFF
                ]
                tasks_signed_off_by_reviewer_count = len(signed_off_reviewed_tasks)
                avg_major_issues_reviewed_signed_off = (
                    float(np.mean([t.major_issues for t in signed_off_reviewed_tasks]))
                    if signed_off_reviewed_tasks
                    else 0.0
                )
                avg_minor_issues_reviewed_signed_off = (
                    float(np.mean([t.minor_issues for t in signed_off_reviewed_tasks]))
                    if signed_off_reviewed_tasks
                    else 0.0
                )
                avg_quality_reviewed_signed_off = (
                    float(np.mean([t.quality_score for t in signed_off_reviewed_tasks]))
                    if signed_off_reviewed_tasks
                    else 0.0
                )
                avg_review_cycles_reviewed_signed_off = (
                    float(
                        np.mean(
                            [t.revision_count + 1 for t in signed_off_reviewed_tasks]
                        )
                    )
                    if signed_off_reviewed_tasks
                    else 0.0
                )
                tasks_reviewed_by_agent = sum(
                    1
                    for t in st.session_state.tasks_final_state
                    if t.reviewer_id == reviewer.id
                )

                agent_performance_data.append(
                    {
                        "Agent ID": reviewer.id,
                        "Type": "Reviewer",
                        "Domain": reviewer.domain_name,
                        "Tasks Created": 0,
                        "Tasks Signed Off (Owned)": 0,
                        "Avg Quality (Signed Off Owned)": f"{0.0:.2f}",
                        "Avg Review Cycles (Signed Off Owned)": f"{0.0:.1f}",
                        "Avg Major Issues (Signed Off Owned)": f"{0.0:.2f}",
                        "Avg Minor Issues (Signed Off Owned)": f"{0.0:.2f}",
                        "Tasks Needing Work (Owned)": 0,
                        "Tasks Reviewed": tasks_reviewed_by_agent,
                        "Tasks Signed Off (Reviewed)": tasks_signed_off_by_reviewer_count,
                        "Avg Quality (Signed Off Reviewed)": f"{avg_quality_reviewed_signed_off:.2f}",
                        "Avg Review Cycles (Signed Off Reviewed)": f"{avg_review_cycles_reviewed_signed_off:.1f}",
                        "Avg Major Issues (Signed Off Reviewed)": f"{avg_major_issues_reviewed_signed_off:.2f}",
                        "Avg Minor Issues (Signed Off Reviewed)": f"{avg_minor_issues_reviewed_signed_off:.2f}",
                    }
                )
        if agent_performance_data:
            # Define column order for clarity
            column_order = [
                "Agent ID",
                "Type",
                "Domain",
                "Tasks Created",
                "Tasks Signed Off (Owned)",
                "Avg Quality (Signed Off Owned)",
                "Avg Review Cycles (Signed Off Owned)",
                "Avg Major Issues (Signed Off Owned)",
                "Avg Minor Issues (Signed Off Owned)",
                "Tasks Needing Work (Owned)",
                "Tasks Reviewed",
                "Tasks Signed Off (Reviewed)",
                "Avg Quality (Signed Off Reviewed)",
                "Avg Review Cycles (Signed Off Reviewed)",
                "Avg Major Issues (Signed Off Reviewed)",
                "Avg Minor Issues (Signed Off Reviewed)",
            ]
            perf_df = pd.DataFrame(agent_performance_data)  # Create DataFrame first
            # Set Agent ID as index *after* creating the full DataFrame
            # This ensures "Agent ID" is a column for reordering if needed, then becomes index.
            # However, for display, it's often better to keep Agent ID as a column if it's part of column_order

            # Ensure all columns in column_order are present
            for col in column_order:
                if col not in perf_df.columns:
                    perf_df[col] = 0  # Or appropriate default

            # Reorder and then set index
            perf_df = perf_df[column_order].set_index("Agent ID")

            st.dataframe(perf_df)
            # Add download button for agent performance
            csv_agent_perf = perf_df.reset_index().to_csv(index=False).encode("utf-8")
            agent_perf_filename = generate_download_filename(
                "agent_performance_summary", "csv", st.session_state
            )
            st.download_button(
                label="Download Agent Performance as CSV",
                data=csv_agent_perf,
                file_name=agent_perf_filename,
                mime="text/csv",
                key="download_agent_perf",
            )
        else:
            st.write("No agent data to display.")

        # Productivity Variance Plot
        st.subheader("Productivity: Tasks Signed Off Histograms")
        st.markdown(
            "Note: Plots can be saved directly using the options menu (⋮) on each chart."
        )
        col1, col2 = st.columns(2)

        # Trainer Productivity Histogram
        if hasattr(st.session_state, "sim_trainers") and st.session_state.sim_trainers:
            trainer_signed_off_counts = [
                sum(
                    1
                    for t in st.session_state.tasks_final_state
                    if t.owner_id == trainer.id and t.status == TaskStatus.SIGNED_OFF
                )
                for trainer in st.session_state.sim_trainers
            ]

            with col1:
                st.markdown("**Trainer Productivity** (Tasks Signed Off - Owned)")
                if trainer_signed_off_counts:
                    source = pd.DataFrame(
                        {"Tasks Signed Off (Owned)": trainer_signed_off_counts}
                    )
                    chart = (
                        alt.Chart(source)
                        .mark_bar()
                        .encode(
                            alt.X(
                                "Tasks Signed Off (Owned):Q",
                                bin=alt.Bin(maxbins=10),
                                title="Tasks Signed Off (Owned)",
                            ),
                            alt.Y("count()", title="Number of Trainers"),
                        )
                        .properties(title="Trainer Output Distribution")
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.write("No tasks signed off by trainers to plot.")
        else:
            with col1:
                st.markdown("**Trainer Productivity**")
                st.write("No trainer data for productivity histogram.")

        # Reviewer Productivity Histogram
        if (
            hasattr(st.session_state, "sim_reviewers")
            and st.session_state.sim_reviewers
        ):
            reviewer_signed_off_counts = [
                sum(
                    1
                    for t in st.session_state.tasks_final_state
                    if t.reviewer_id == reviewer.id
                    and t.status == TaskStatus.SIGNED_OFF
                )
                for reviewer in st.session_state.sim_reviewers
            ]

            with col2:
                st.markdown(
                    "**Reviewer Productivity** (Tasks Signed Off - Reviewed by them)"
                )
                if reviewer_signed_off_counts:
                    source_rev = pd.DataFrame(
                        {"Tasks Signed Off (Reviewed)": reviewer_signed_off_counts}
                    )
                    chart_rev = (
                        alt.Chart(source_rev)
                        .mark_bar()
                        .encode(
                            alt.X(
                                "Tasks Signed Off (Reviewed):Q",
                                bin=alt.Bin(maxbins=10),
                                title="Tasks Signed Off (Reviewed)",
                            ),
                            alt.Y("count()", title="Number of Reviewers"),
                        )
                        .properties(title="Reviewer Output Distribution")
                    )
                    st.altair_chart(chart_rev, use_container_width=True)
                else:
                    st.write("No tasks signed off by reviewers to plot.")
        else:
            with col2:
                st.markdown("**Reviewer Productivity**")
                st.write("No reviewer data for productivity histogram.")

    elif (
        hasattr(st.session_state, "tasks_final_state")
        and not st.session_state.tasks_final_state
    ):
        st.write("No tasks were created or processed in the simulation.")

elif not st.session_state.simulation_run:
    st.info(
        "Adjust parameters in the sidebar and click 'Run Simulation' to see results."
    )
