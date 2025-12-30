# PersonaNetworkSimulator

A research and simulation framework for studying the emergence of apparent consensus from synthetic personas in controlled environments.

**⚠️ WARNING: This tool is designed for platform safety research, NOT deployment. Do not use to create or deploy synthetic persona networks on real platforms.**

## Purpose

PersonaNetworkSimulator helps researchers understand and detect synthetic persona farms by:

1. Creating synthetic agents with stable persona profiles, linguistic style constraints, and bounded memory
2. Simulating multi-agent interaction over time
3. Tracking narrative convergence, divergence, and reinforcement
4. Producing detection signals useful for moderation systems

## Installation

```bash
# Clone the repository
cd synthetic-personal-farm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Quick Start

### Running a Demo Simulation

```bash
# Run with default settings
python -m persona_sim.cli demo

# Run with custom parameters
python -m persona_sim.cli demo --personas 50 --coordinated-fraction 0.3 --hours 48 --seed 42

# Save output to a directory
python -m persona_sim.cli demo -o ./results
```

### Using as a Library

```python
from persona_sim.agents.persona import Persona, PersonaProfile
from persona_sim.simulation.engine import SimulationEngine, SimulationConfig
from persona_sim.analysis.independence import IndependenceDetector

# Create personas
profile = PersonaProfile(
    name="Alice",
    age=28,
    occupation="Engineer",
    interests=["technology", "environment"],
)
persona = Persona(profile)
persona.add_belief("climate_action", position=0.7, confidence=0.6)

# Create simulation
config = SimulationConfig(
    duration_hours=24.0,
    seed=42,
)
engine = SimulationEngine(config)
engine.add_persona(persona)

# Run simulation
state = engine.run()

# Analyze for coordination
detector = IndependenceDetector()
events = engine.get_all_events()
signals = detector.analyze_timing_correlation(events, [persona.id])
```

## Repository Structure

```
persona_sim/
├── agents/
│   ├── persona.py        # Persona traits & constraints
│   ├── memory.py         # Rolling memory summaries
│   └── style.py          # Linguistic fingerprints
├── network/
│   ├── graph.py          # Social graph (who talks to whom)
│   └── dynamics.py       # Interaction scheduling
├── simulation/
│   ├── engine.py         # Time-step simulation loop
│   └── narratives.py     # Topic & belief tracking
├── analysis/
│   ├── convergence.py    # Measure consensus formation
│   ├── independence.py   # Detect coordination signals
│   └── metrics.py        # Simulation metrics
├── visualization/
│   └── plots.py          # Plotting utilities
├── cli.py                # Command-line interface
└── tests/                # Test suite
```

## Key Concepts

### Personas

Synthetic agents with:
- **Stable profiles**: Name, age, occupation, interests, political leaning
- **Personality traits**: Big Five model (openness, conscientiousness, etc.)
- **Beliefs**: Positions on topics with confidence levels
- **Susceptibility**: How easily influenced by others
- **Credibility**: How much others trust this persona

### Memory System

Bounded, lossy memory that:
- Stores recent detailed interactions
- Summarizes older memories by topic
- Computes fingerprints for similarity detection

### Linguistic Style

Each persona has consistent style constraints:
- Tone level (formal to informal)
- Vocabulary complexity
- Use of contractions, emoji
- Favorite phrases

### Network Dynamics

- **Social graph**: Directed connections between personas
- **Interaction scheduling**: Natural timing with variance
- **Burst activity**: Rapid successive posts
- **Coordinated responses**: For testing detection

### Analysis Tools

1. **Convergence Analyzer**: Measures how beliefs converge over time
2. **Independence Detector**: Identifies coordination signals:
   - Timing correlation (synchronized posting)
   - Phrasing similarity
   - Belief update synchronicity
   - Response patterns

## Ethical Guardrails

This framework is designed with safety in mind:

1. **No social platform integration**: Agents cannot access external APIs or platforms
2. **Rule-based generation**: All language generation is templated, not AI-generated
3. **No real identity modeling**: Does not model real people
4. **Clear documentation**: Misuse risks are documented
5. **Detection focus**: Primary goal is to help detect, not create, persona farms

## Running Tests

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest persona_sim/tests/

# Run with verbose output
pytest -v persona_sim/tests/

# Run specific test file
pytest persona_sim/tests/test_agents.py
```

## Research Applications

- **Platform safety research**: Understanding how coordinated inauthentic behavior emerges
- **Detection algorithm development**: Testing coordination detection methods
- **Moderation tool development**: Training classifiers on synthetic data
- **Academic research**: Studying information cascade dynamics

## Outputs

The simulator produces:

- **Narrative convergence graphs**: How beliefs change over time
- **Persona similarity matrices**: Pairwise belief similarity
- **Coordination signals**: Detected patterns of synthetic consensus
- **Metrics reports**: Comprehensive simulation statistics

## Contributing

Contributions are welcome! Please ensure:

1. All code follows the existing style
2. Tests are included for new features
3. Documentation is updated
4. Ethical guidelines are followed

## License

This project is for research purposes only. See LICENSE for details.

## Disclaimer

This software is provided for legitimate research purposes only. The authors are not responsible for any misuse. Users are responsible for ensuring their use complies with applicable laws and platform policies.
