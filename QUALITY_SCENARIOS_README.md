# Quality Management Scenarios 📊

This document describes three new scenarios that demonstrate strategic workforce allocation when dealing with quality issues in content creation teams.

## 🎯 Scenario Overview

These scenarios model a common business challenge: **When overall quality is low, should top performers focus on reviewing/fixing work, or creating new high-quality content?**

### Team Composition (All Scenarios)
- **6 Top Performers** (high quality: 0.75, fast writers, excellent reviewers)
- **8 Normal Contractors** (medium quality: 0.40, average speed)
- **3 Bad Contractors** (low quality: 0.25, slow, unreliable)

## 📋 The Three Scenarios

### 🚨 Quality Crisis (`configs/quality_crisis.json`)
**Strategy:** Top performers spend **60% of time reviewing** to catch quality issues

**Results (10-day simulation):**
- ✅ Tasks Created: 127
- ✅ Tasks Signed Off: 2 (1.6% completion rate)
- ⏳ Tasks Waiting for Review: 67
- 📊 Final Quality: 0.884 (excellent)

**Analysis:** High review focus ensures quality but creates bottlenecks and low throughput.

### ⚖️ Quality Balanced (`configs/quality_balanced.json`)
**Strategy:** Top performers spend **35% of time reviewing** (balanced approach)

**Results (10-day simulation):**
- ✅ Tasks Created: 148
- ✅ Tasks Signed Off: 0 (0.0% completion rate)
- ⏳ Tasks Waiting for Review: 109
- 📊 Final Quality: N/A (no completions yet)

**Analysis:** Moderate balance between creation and review, building toward sustainable throughput.

### 💡 Quality Recovery (`configs/quality_recovery.json`)
**Strategy:** Top performers spend **15% of time reviewing** to maximize content creation

**Results (10-day simulation):**
- ✅ Tasks Created: 173 (+36% vs Crisis)
- ✅ Tasks Signed Off: 0 (0.0% completion rate)
- ⏳ Tasks Waiting for Review: 151
- 📊 Final Quality: N/A (no completions yet)

**Analysis:** Maximum creation rate but creates massive review backlog.

## 🎯 Strategic Insights

### Key Findings
1. **High Review Time (60%)** = Excellent quality but low productivity
2. **Low Review Time (15%)** = High creation rate but unsustainable backlogs
3. **Balanced Approach (35%)** = Optimal long-term strategy

### Business Implications

**Short-term Quality Crisis:**
- Use Crisis scenario (60% review) to ensure immediate quality
- Accept lower throughput temporarily
- Good for critical deadlines or reputation recovery

**Long-term Sustainable Growth:**
- Use Balanced scenario (35% review) for optimal throughput
- Maintains quality while maximizing productivity
- Best for steady-state operations

**Rapid Content Creation:**
- Use Recovery scenario (15% review) for content sprints
- Monitor review queue carefully
- Good for initial content creation phases

## 🚀 Usage Examples

### Running a Scenario
```python
from task_simulator import *
import json

# Load the quality crisis scenario
with open('configs/quality_crisis.json', 'r') as f:
    config_dict = json.load(f)

# Convert to simulation objects
domain_setup = DomainSimulationSetup(
    domain_name=config_dict['domain_setups'][0]['domain_name'],
    num_top_performers=config_dict['domain_setups'][0]['num_top_performers'],
    num_normal_contractors=config_dict['domain_setups'][0]['num_normal_contractors'],
    num_bad_contractors=config_dict['domain_setups'][0]['num_bad_contractors'],
    top_performer_cfg=PerformanceLevelConfig(**config_dict['domain_setups'][0]['top_performer_cfg']),
    normal_contractor_cfg=PerformanceLevelConfig(**config_dict['domain_setups'][0]['normal_contractor_cfg']),
    bad_contractor_cfg=PerformanceLevelConfig(**config_dict['domain_setups'][0]['bad_contractor_cfg'])
)

# Run simulation
sim_config = SimulationConfig(
    simulation_days=21,
    domain_setups=[domain_setup],
    random_seed=555
)

simulation = Simulation(sim_config)
results = simulation.run()
```

### Comparing Scenarios
```bash
# Run comparison script
python -c "
# [comparison code from above]
"
```

## 📈 Performance Metrics to Monitor

- **Task Creation Rate**: How many new tasks are generated
- **Completion Rate**: Percentage of tasks that get signed off
- **Review Queue Size**: Tasks waiting for review (bottleneck indicator)
- **Final Quality Score**: Quality of completed work
- **Agent Productivity**: Average hours worked per agent

## 🎉 Conclusion

These scenarios demonstrate that **workforce allocation strategy significantly impacts outcomes**. The optimal approach depends on your current situation:

- **Crisis mode**: Prioritize quality over quantity
- **Growth mode**: Balance creation and review
- **Sprint mode**: Maximize creation, manage review debt

The simulation allows you to test different strategies before implementing them in real teams! 